"""Lifecycle tests for qf.engine against a fake broker.

Each engine invariant (module docstring of qf/engine.py) has at least one
test here, named for the invariant it protects. The fake broker is a
small in-memory exchange: market orders fill instantly at a settable
price, OCOs sit "live" until a test triggers them, and any method can be
told to fail.

Numbers used throughout: bars with a 5-point range on a 100 close give
ATR1h = 5%, so k=2 -> target 10%, stop 5% (comfortably above the fee
floor). A $1,000 buy at 100 fills 10 SOL with a 0.035 SOL fee, leaving
9.965 sellable.

Run with: pytest qf/tests -m unit
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from qf import state
from qf.broker import AlgoStatus, BrokerError
from qf.engine import FeeFloorError, QFError, make_trade_id
from qf.tests.conftest import SYM, T0, _bars
from vt.data.feed import Quote
from vt.journal import store as journal

pytestmark = pytest.mark.unit


def _open(qf, **kw):
    return qf.open(SYM, kw.pop("lane", "small"), manual=True, **kw)


# --------------------------------------------------------------------------- #
# open
# --------------------------------------------------------------------------- #


def test_open_places_tagged_entry_and_oco_sized_net_of_fee(rig):
    qf, broker, _ = rig
    pos = _open(qf)

    assert pos.size == pytest.approx(9.965)
    assert (pos.entry_px, pos.target_px, pos.stop_px, pos.arm_px) == (100.0, 110.0, 95.0, 105.0)
    buy = next(c for c in broker.calls if c[0] == "market_buy_usd")
    oco = next(c for c in broker.calls if c[0] == "place_oco")
    assert buy[2] == 1_000.0 and buy[3] == f"{pos.trade_id}e"
    assert oco[1:] == (SYM, pytest.approx(9.965), 110.0, 95.0, f"{pos.trade_id}p")
    assert pos.trade_id.startswith("qfSOL") and pos.trade_id.isalnum()
    assert qf.book().positions == (pos,)
    assert pos.deadline == T0 + timedelta(hours=8)


def test_invariant_thesis_is_journaled_before_the_entry_order(rig):
    qf, broker, _ = rig
    seen = []
    broker.on_buy = lambda: seen.append(len(journal.read_cards(qf.cfg.journal_path)))
    _open(qf)
    assert seen == [1]
    card = journal.read_cards(qf.cfg.journal_path)[0]
    assert card["manual"] is True and card["strategy"] == "quickflip"


def test_invariant_demo_only(rig):
    qf, broker, _ = rig
    broker.is_demo = False
    with pytest.raises(QFError, match="not demo"):
        _open(qf)
    assert "market_buy_usd" not in broker.names()


def test_automated_entry_is_refused_until_a_signal_exists(rig):
    qf, broker, _ = rig
    with pytest.raises(QFError, match="signal filter"):
        qf.open(SYM, "small", manual=False)


def test_fee_floor_blocks_open_without_touching_broker_or_journal(rig):
    qf, broker, _ = rig
    qf._get_bars = lambda s, t, limit: _bars(s, t, limit=limit, rng=1.0)  # ATR 1% -> fails floor
    with pytest.raises(FeeFloorError, match="fee floor"):
        _open(qf)
    assert "market_buy_usd" not in broker.names()
    assert journal.read_cards(qf.cfg.journal_path) == []


def test_fee_floor_override_opens_and_flags_the_card(rig):
    qf, _, _ = rig
    qf._get_bars = lambda s, t, limit: _bars(s, t, limit=limit, rng=1.0)
    _open(qf, allow_below_floor=True)
    assert journal.read_cards(qf.cfg.journal_path)[0]["below_fee_floor_override"] is True


def test_one_position_per_symbol(rig):
    qf, _, _ = rig
    _open(qf)
    with pytest.raises(QFError, match="one per symbol"):
        _open(qf)


def test_bankroll_cap(rig):
    qf, _, _ = rig
    qf.cfg = replace(qf.cfg, max_open_notional_usd=1_500.0)
    _open(qf)
    with pytest.raises(QFError, match="bankroll cap"):
        qf.open("ETH-USDT", "small", manual=True)


def test_concurrency_cap(rig):
    qf, _, _ = rig
    qf.cfg = replace(qf.cfg, max_concurrent_positions=1)
    _open(qf)
    with pytest.raises(QFError, match="concurrency cap"):
        qf.open("ETH-USDT", "small", manual=True)


def test_halted_book_refuses_opens(rig):
    qf, _, _ = rig
    state.save(qf.cfg.state_path, qf.book().halt("test halt"))
    with pytest.raises(QFError, match="halted"):
        _open(qf)


def test_entry_failure_is_journaled_and_raised(rig):
    qf, broker, _ = rig
    broker.fail.add("market_buy_usd")
    with pytest.raises(QFError, match="entry failed"):
        _open(qf)
    card = journal.read_cards(qf.cfg.journal_path)[0]
    assert card["outcome"]["exit_reason"] == "entry_failed"
    assert qf.book().positions == ()


def test_invariant_never_unprotected_oco_failure_flattens_and_halts(rig):
    qf, broker, _ = rig
    broker.fail.add("place_oco")
    with pytest.raises(QFError, match="could not protect"):
        _open(qf)

    sell = next(c for c in broker.calls if c[0] == "market_sell")
    assert sell[2] == pytest.approx(9.965) and sell[3] == f"{sell[3][:-2]}x0"
    book = qf.book()
    assert book.positions == () and book.halted
    assert journal.closed_cards(qf.cfg.journal_path)[0]["outcome"]["exit_reason"] == "protect_failed"


def test_oco_and_flatten_both_failing_keeps_position_on_book_loudly(rig):
    qf, broker, _ = rig
    broker.fail |= {"place_oco", "market_sell"}
    with pytest.raises(QFError):
        _open(qf)
    book = qf.book()
    assert len(book.positions) == 1 and book.positions[0].algo_id is None
    assert book.halted and any("UNPROTECTED" in a for a in book.alerts)


def test_failed_balance_read_does_not_block_protection(rig):
    qf, broker, _ = rig
    broker.fail.add("holding")
    pos = _open(qf)
    assert pos.algo_id is not None and pos.size == pytest.approx(9.965)


# --------------------------------------------------------------------------- #
# tick -- exchange-side exits
# --------------------------------------------------------------------------- #


def test_target_fill_closes_with_net_pnl(rig):
    qf, broker, clock = rig
    pos = _open(qf)
    clock.now += timedelta(hours=3)
    broker.trigger(pos.algo_id, "tp", 110.0, pos.size)

    events = qf.tick()

    assert "target" in events[0]
    assert qf.book().positions == ()
    out = journal.closed_cards(qf.cfg.journal_path)[0]["outcome"]
    assert out["exit_reason"] == "target"
    assert out["pnl_usd"] == pytest.approx(9.965 * 110 * (1 - 0.0035) - 1_000.0)
    assert out["time_in_trade_minutes"] == 180.0


def test_stop_after_arm_is_labelled_breakeven(rig):
    qf, broker, _ = rig
    pos = _open(qf)
    state.save(qf.cfg.state_path, qf.book().with_position(replace(pos, armed=True)))
    broker.trigger(pos.algo_id, "sl", 100.71, pos.size)
    qf.tick()
    assert journal.closed_cards(qf.cfg.journal_path)[0]["outcome"]["exit_reason"] == "stop_breakeven"


def test_externally_cancelled_oco_flattens_and_halts(rig):
    qf, broker, _ = rig
    pos = _open(qf)
    broker.algos[pos.algo_id] = AlgoStatus("canceled", None, None)
    qf.tick()
    assert qf.book().halted and qf.book().positions == ()
    assert journal.closed_cards(qf.cfg.journal_path)[0]["outcome"]["exit_reason"] == "protect_failed"


# --------------------------------------------------------------------------- #
# tick -- breakeven arm
# --------------------------------------------------------------------------- #


def test_hold_updates_watermarks_only(rig):
    qf, broker, clock = rig
    _open(qf)
    broker.px = 103.0
    clock.now += timedelta(minutes=5)
    assert qf.tick() == []
    pos = qf.book().positions[0]
    assert (pos.high_px, pos.armed) == (103.0, False)


def test_invariant_arm_tightens_stop_to_net_breakeven(rig):
    qf, broker, clock = rig
    pos = _open(qf)
    broker.px = 105.0
    clock.now += timedelta(hours=1)
    qf.tick()

    assert ("amend_stop", SYM, pos.algo_id, 100.71) in broker.calls
    armed = qf.book().positions[0]
    assert armed.armed and armed.stop_px == 100.71 and armed.algo_id == pos.algo_id


def test_arm_falls_back_to_cancel_and_replace(rig):
    qf, broker, clock = rig
    pos = _open(qf)
    broker.fail.add("amend_stop")
    broker.px = 106.0
    clock.now += timedelta(hours=1)
    qf.tick()

    armed = qf.book().positions[0]
    assert armed.armed and armed.algo_id != pos.algo_id
    replace_call = [c for c in broker.calls if c[0] == "place_oco"][-1]
    assert replace_call[3:] == (110.0, 100.71, f"{pos.trade_id}a")
    assert broker.names().index("cancel_algo") < len(broker.names()) - 1


def test_arm_with_amend_and_replace_failing_flattens(rig):
    qf, broker, clock = rig
    _open(qf)
    broker.fail.add("amend_stop")
    broker.px = 106.0
    clock.now += timedelta(hours=1)
    broker.fail.add("place_oco")
    qf.tick()
    assert qf.book().positions == () and qf.book().halted


def test_stale_quote_skips_arming(rig):
    qf, broker, clock = rig
    _open(qf)
    broker.px = 106.0
    qf._get_quote = lambda s: Quote(s, 105.9, 106.0, 106.0, clock() - timedelta(minutes=1), "okx_demo")
    clock.now += timedelta(hours=1)
    qf.tick()
    assert qf.book().positions[0].armed is False


# --------------------------------------------------------------------------- #
# tick -- time-stop
# --------------------------------------------------------------------------- #


def test_invariant_time_exit_cancels_oco_then_sells_only_recorded_size(rig):
    qf, broker, clock = rig
    pos = _open(qf)
    clock.now = pos.deadline
    qf.tick()

    names = broker.names()
    assert names.index("cancel_algo") < len(names) - 1 and names[-1] == "market_sell"
    assert broker.calls[-1][2] == pytest.approx(9.965)  # not the 1,000,000 SOL the account holds
    assert journal.closed_cards(qf.cfg.journal_path)[0]["outcome"]["exit_reason"] == "time"


def test_time_exit_needs_no_fresh_price(rig):
    qf, broker, clock = rig
    pos = _open(qf)
    qf._get_quote = lambda s: Quote(s, None, None, None, clock() - timedelta(hours=1), "okx_demo")
    clock.now = pos.deadline + timedelta(minutes=1)
    qf.tick()
    assert qf.book().positions == ()


def test_time_exit_racing_a_triggered_oco_records_the_oco_fill(rig):
    qf, broker, clock = rig
    pos = _open(qf)
    broker.trigger(pos.algo_id, "sl", 95.0, pos.size)
    broker.fail.add("cancel_algo")
    # algo_status is checked first on the tick, so force the race path:
    broker.algos[pos.algo_id] = AlgoStatus("live", None, None)
    real_status = broker.algo_status
    calls = {"n": 0}

    def racing_status(aid):
        calls["n"] += 1
        return real_status(aid) if calls["n"] == 1 else AlgoStatus("effective", list(broker.orders)[-1], "sl")

    broker.algo_status = racing_status
    clock.now = pos.deadline
    qf.tick()

    assert "market_sell" not in broker.names()
    assert journal.closed_cards(qf.cfg.journal_path)[0]["outcome"]["exit_reason"] == "stop"


def test_failed_time_exit_sell_is_retried_next_tick(rig):
    qf, broker, clock = rig
    pos = _open(qf)
    clock.now = pos.deadline
    broker.fail.add("market_sell")
    qf.tick()

    book = qf.book()
    assert book.halted and book.positions[0].algo_id is None
    assert any("UNPROTECTED" in a for a in book.alerts)

    broker.fail.clear()
    qf.tick()
    assert qf.book().positions == ()
    assert broker.names().count("cancel_algo") == 1  # not re-cancelled


# --------------------------------------------------------------------------- #
# breaker / reconcile / kill / resume
# --------------------------------------------------------------------------- #


def test_invariant_consecutive_losses_halt(rig):
    qf, broker, clock = rig
    for i in range(3):
        clock.now = T0 + timedelta(minutes=i)
        pos = _open(qf) if i == 0 else qf.open(SYM, "small", manual=True)
        broker.trigger(pos.algo_id, "sl", 95.0, pos.size)
        qf.tick()
    book = qf.book()
    assert book.consecutive_losses == 3 and book.halted
    with pytest.raises(QFError, match="halted"):
        _open(qf)


def test_a_win_resets_the_loss_streak(rig):
    qf, broker, _ = rig
    state.save(qf.cfg.state_path, replace(qf.book(), consecutive_losses=2))
    pos = _open(qf)
    broker.trigger(pos.algo_id, "tp", 110.0, pos.size)
    qf.tick()
    assert qf.book().consecutive_losses == 0 and not qf.book().halted


def test_reconcile_halts_only_when_coins_are_missing(rig):
    qf, broker, _ = rig
    _open(qf)
    assert qf.reconcile() == []

    broker.holdings["SOL"] = 5.0
    drifts = qf.reconcile()
    assert len(drifts) == 1 and qf.book().halted


def test_kill_sells_recorded_sizes_and_sweeps_tagged_algos(rig):
    qf, broker, _ = rig
    pos = _open(qf)
    orphan = broker.place_oco("ETH-USDT", 1.0, 5_000.0, 4_000.0, "qfETHorphan")

    report = qf.kill()

    assert report.closed == (pos.trade_id,)
    assert set(report.cancelled_algos) == {pos.algo_id, orphan}
    assert broker.calls[[c[0] for c in broker.calls].index("market_sell")][2] == pytest.approx(9.965)
    assert qf.book().halted and qf.book().positions == ()
    assert journal.closed_cards(qf.cfg.journal_path)[0]["outcome"]["exit_reason"] == "kill"


def test_kill_dry_run_touches_nothing(rig):
    qf, broker, _ = rig
    _open(qf)
    before = len(broker.calls)
    report = qf.kill(dry_run=True)
    assert len(report.closed) == 1
    assert [c[0] for c in broker.calls[before:]] == ["tagged_live_algos"]  # a read only
    assert not qf.book().halted


def test_resume_clears_halt_alerts_and_streak(rig):
    qf, _, _ = rig
    state.save(qf.cfg.state_path, replace(qf.book().halt("x"), consecutive_losses=3))
    book = qf.resume()
    assert not book.halted and book.alerts == () and book.consecutive_losses == 0


def test_closed_summary_splits_manual_from_signal(rig):
    qf, broker, _ = rig
    pos = _open(qf)
    broker.trigger(pos.algo_id, "tp", 110.0, pos.size)
    qf.tick()
    summary = qf.closed_summary()
    assert summary["manual"]["n"] == 1 and summary["manual"]["wins"] == 1
    assert summary["signal"] == {"n": 0, "wins": 0, "net_pnl_usd": 0, "expectancy_r": None}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def test_trade_id_fits_okx_client_id_rules():
    tid = make_trade_id("qf", "1INCH-USDT", T0)
    assert tid == "qf1INCH260925120030" and tid.isalnum() and len(tid) + 1 <= 32


def test_fee_model_falls_back_when_fee_read_fails(rig):
    qf, broker, _ = rig
    broker.taker_rate = lambda s: (_ for _ in ()).throw(BrokerError("down"))
    assert qf.fee_model(SYM).taker_rate == qf.cfg.fallback_taker_rate
