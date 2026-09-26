"""Crash-recovery and race tests for qf.engine (invariants 7 and 8, plus
the kill/entry races a code review found).

A "crash" here is simulated by writing the book the way it would look if
the process had died mid-step, then running the next tick against it --
the property under test is always "no second sell, nothing orphaned,
nothing silently lost".

Run with: pytest qf/tests -m unit
"""

from __future__ import annotations

import threading
from dataclasses import replace
from datetime import timedelta

import pytest

from qf import state
from qf.broker import AlgoStatus
from qf.engine import QFError
from qf.state import StateError
from qf.tests.conftest import SYM
from vt.journal import store as journal

pytestmark = pytest.mark.unit


def _open(qf):
    return qf.open(SYM, "small", manual=True)


def _sells(broker):
    return [c for c in broker.calls if c[0] == "market_sell"]


# --------------------------------------------------------------------------- #
# Invariant 7 -- restart-safe exits
# --------------------------------------------------------------------------- #


def test_crash_after_journal_before_save_drops_the_position_without_reselling(rig):
    qf, broker, _ = rig
    pos = _open(qf)
    broker.trigger(pos.algo_id, "tp", 110.0, pos.size)
    book_before = qf.book()
    qf.tick()  # journals the close...
    state.save(qf.cfg.state_path, book_before)  # ...then "crashes" before the book save lands

    events = qf.tick()

    assert "already journaled" in events[0]
    assert qf.book().positions == ()
    assert _sells(broker) == []
    assert len(journal.closed_cards(qf.cfg.journal_path)) == 1


def test_duplicate_outcome_is_treated_as_already_closed(rig):
    qf, broker, _ = rig
    pos = _open(qf)
    broker.trigger(pos.algo_id, "tp", 110.0, pos.size)
    journal.patch_outcome(pos.trade_id, {"r_multiple": 1.0, "exit_reason": "target"}, path=qf.cfg.journal_path)

    book = qf._record_close(qf.book(), pos, "target", broker.orders[broker.algos[pos.algo_id].ord_id])

    assert book.positions == ()
    assert any("already journaled" in a for a in book.alerts)


def test_exit_checkpoint_reads_the_existing_sell_instead_of_selling_again(rig):
    qf, broker, clock = rig
    pos = _open(qf)
    clock.now = pos.deadline
    broker.sell_state = "live"  # the sell is accepted but not yet filled when we "crash"
    qf.tick()

    mid = qf.book().positions[0]
    assert mid.exit_order_id is not None and mid.algo_id is None
    assert len(_sells(broker)) == 1

    broker.orders[mid.exit_order_id] = replace(broker.orders[mid.exit_order_id], state="filled", filled_sz=pos.size)
    qf.tick()

    assert qf.book().positions == ()
    assert len(_sells(broker)) == 1  # the restart read the checkpointed fill
    assert journal.closed_cards(qf.cfg.journal_path)[0]["outcome"]["exit_reason"] == "time"


def test_sell_that_died_unfilled_is_retried_with_a_fresh_client_id(rig):
    qf, broker, clock = rig
    pos = _open(qf)
    clock.now = pos.deadline
    broker.sell_state = "canceled"
    qf.tick()
    assert qf.book().positions[0].exit_order_id is None
    assert qf.book().halted

    broker.sell_state = "filled"
    qf.tick()

    ids = [c[3] for c in _sells(broker)]
    assert ids == [f"{pos.trade_id}t0", f"{pos.trade_id}t1"]
    assert qf.book().positions == ()


def test_failed_flatten_before_deadline_is_retried_every_tick(rig):
    qf, broker, clock = rig
    broker.fail |= {"place_oco", "market_sell"}
    with pytest.raises(QFError):
        _open(qf)
    assert qf.book().positions[0].exit_reason == "protect_failed"

    broker.fail.clear()
    clock.now += timedelta(minutes=1)  # long before the 8h deadline
    qf.tick()

    assert qf.book().positions == ()
    assert journal.closed_cards(qf.cfg.journal_path)[0]["outcome"]["exit_reason"] == "protect_failed"


def test_orphaned_replacement_oco_is_cancelled_before_selling(rig):
    qf, broker, clock = rig
    pos = _open(qf)
    # crash window in _arm: old OCO cancelled + checkpointed, replacement
    # placed on the exchange, but the new algo id never reached the book
    broker.cancel_algo(SYM, pos.algo_id)
    orphan = broker.place_oco(SYM, pos.size, pos.target_px, pos.breakeven_px, f"{pos.trade_id}a")
    state.save(qf.cfg.state_path, qf.book().with_position(replace(pos, algo_id=None)))

    clock.now += timedelta(minutes=5)
    qf.tick()

    names = broker.names()
    assert ("cancel_algo", SYM, orphan) in broker.calls
    assert names.index("market_sell") > max(i for i, c in enumerate(broker.calls) if c == ("cancel_algo", SYM, orphan))
    assert qf.book().positions == ()


# --------------------------------------------------------------------------- #
# Kill races
# --------------------------------------------------------------------------- #


def test_kill_racing_a_triggered_oco_records_its_fill_and_does_not_sell(rig):
    qf, broker, _ = rig
    pos = _open(qf)
    broker.trigger(pos.algo_id, "sl", 95.0, pos.size)
    broker.fail.add("cancel_algo")

    report = qf.kill()

    assert report.closed == (pos.trade_id,)
    assert _sells(broker) == []
    assert journal.closed_cards(qf.cfg.journal_path)[0]["outcome"]["exit_reason"] == "stop"
    assert qf.book().positions == ()


def test_kill_never_sells_under_an_oco_it_could_not_cancel(rig):
    qf, broker, _ = rig
    pos = _open(qf)
    broker.fail.add("cancel_algo")

    report = qf.kill()

    assert report.closed == ()
    assert any("NOT sold" in e for e in report.errors)
    assert _sells(broker) == []
    assert qf.book().positions[0].trade_id == pos.trade_id and qf.book().halted


# --------------------------------------------------------------------------- #
# Entry that is still filling
# --------------------------------------------------------------------------- #


def test_entry_still_filling_is_cancelled_then_protected_at_final_size(rig):
    qf, broker, _ = rig
    broker.buy_states = ["partially_filled"]
    pos = _open(qf)

    assert "cancel_order" in broker.names()
    assert pos.algo_id is not None and pos.size == pytest.approx(9.965)


def test_entry_that_cannot_be_stopped_is_flattened_and_halts(rig):
    qf, broker, _ = rig
    broker.buy_states = ["partially_filled"]
    broker.fail.add("cancel_order")  # can't stop it: more coins could still arrive

    with pytest.raises(QFError, match="more may fill unmanaged"):
        _open(qf)

    assert "place_oco" not in broker.names()
    assert len(_sells(broker)) == 1
    assert qf.book().halted and qf.book().positions == ()


def test_missing_actual_side_is_inferred_from_fill_price(rig):
    qf, broker, _ = rig
    pos = _open(qf)
    broker.trigger(pos.algo_id, "tp", 110.0, pos.size)
    st = broker.algos[pos.algo_id]
    broker.algos[pos.algo_id] = AlgoStatus(st.state, st.ord_id, None)
    qf.tick()
    assert journal.closed_cards(qf.cfg.journal_path)[0]["outcome"]["exit_reason"] == "target"


# --------------------------------------------------------------------------- #
# Reconcile: crash inside open
# --------------------------------------------------------------------------- #


def test_reconcile_flags_a_journaled_entry_with_no_book_row(rig):
    qf, _, _ = rig
    pos = _open(qf)
    state.save(qf.cfg.state_path, qf.book().without(pos.trade_id))  # crashed before the book save

    drifts = qf.reconcile()

    assert len(drifts) == 1 and "untracked fill" in drifts[0]
    assert qf.book().halted


def test_reconcile_skips_positions_mid_exit(rig):
    qf, broker, _ = rig
    pos = _open(qf)
    state.save(qf.cfg.state_path, qf.book().with_position(replace(pos, algo_id=None, exit_order_id="o99")))
    broker.holdings["SOL"] = 0.0
    assert qf.reconcile() == []


# --------------------------------------------------------------------------- #
# Invariant 8 -- one writer
# --------------------------------------------------------------------------- #


def test_lock_blocks_a_second_writer_until_released(tmp_path):
    path = tmp_path / "state.json"
    with state.locked(path, timeout_seconds=1.0):
        with pytest.raises(StateError, match="locked"):
            with state.locked(path, timeout_seconds=0.3):
                pass
    with state.locked(path, timeout_seconds=0.3):  # released cleanly
        pass
    assert not path.with_suffix(".json.lock").exists()


def test_lock_serialises_concurrent_writers(tmp_path):
    path = tmp_path / "state.json"
    state.save(path, state.Book())
    errors: list[Exception] = []

    def bump() -> None:
        try:
            for _ in range(20):
                with state.locked(path, timeout_seconds=10.0):
                    book = state.load(path)
                    state.save(path, replace(book, consecutive_losses=book.consecutive_losses + 1))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=bump) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert state.load(path).consecutive_losses == 60  # no lost update
