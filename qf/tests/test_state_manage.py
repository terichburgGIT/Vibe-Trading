"""Tests for qf.state (persisted book) and qf.manage (pure exit logic).

Run with: pytest qf/tests -m unit
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from qf import manage, state
from qf.manage import Action
from qf.state import Book, Position, StateError

pytestmark = pytest.mark.unit

T0 = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)


def _pos(**overrides) -> Position:
    base = dict(
        trade_id="qfSOL260925120000",
        symbol="SOL-USDT",
        lane="small",
        manual=True,
        notional_usd=1_000.0,
        size=9.965,  # 10 SOL filled @ 100, 0.35% fee taken in SOL
        entry_px=100.0,
        target_px=102.0,
        stop_px=99.0,
        breakeven_px=100.71,
        arm_px=101.0,
        planned_risk_usd=19.0,
        opened_at=T0,
        deadline=T0 + timedelta(hours=8),
        algo_id="algo-1",
    )
    base.update(overrides)
    return Position(**base)


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #


def test_missing_state_file_is_an_empty_book(tmp_path):
    assert state.load(tmp_path / "nope.json") == Book()


def test_book_round_trips_through_disk(tmp_path):
    path = tmp_path / "state.json"
    book = Book(consecutive_losses=2).with_position(_pos(armed=True, high_px=101.5, low_px=99.4))
    state.save(path, book)

    assert state.load(path) == book
    assert not path.with_suffix(".json.tmp").exists()


def test_corrupt_state_raises_instead_of_reading_empty(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(StateError):
        state.load(path)


def test_position_ignores_unknown_keys_from_newer_writers(tmp_path):
    path = tmp_path / "state.json"
    state.save(path, Book().with_position(_pos()))
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["positions"][0]["future_field"] = 1
    path.write_text(json.dumps(raw), encoding="utf-8")

    assert state.load(path).positions[0] == _pos()


def test_with_position_replaces_by_trade_id():
    book = Book().with_position(_pos()).with_position(_pos(armed=True))
    assert len(book.positions) == 1
    assert book.positions[0].armed is True


def test_halt_records_reason_and_alert():
    book = Book().halt("3 consecutive losses")
    assert book.halted and book.halt_reason == "3 consecutive losses"
    assert book.alerts == ("3 consecutive losses",)


# --------------------------------------------------------------------------- #
# manage.decide / observe
# --------------------------------------------------------------------------- #


def test_hold_between_levels():
    assert manage.decide(_pos(), 100.5, T0 + timedelta(hours=1)) is Action.HOLD


def test_arm_once_price_reaches_arm_level():
    assert manage.decide(_pos(), 101.0, T0 + timedelta(hours=1)) is Action.ARM


def test_never_re_arms():
    assert manage.decide(_pos(armed=True), 101.8, T0 + timedelta(hours=1)) is Action.HOLD


def test_time_exit_outranks_arm():
    assert manage.decide(_pos(), 101.5, T0 + timedelta(hours=8)) is Action.TIME_EXIT


def test_armed_stop_never_loosens():
    assert manage.armed_stop(_pos()) == 100.71
    assert manage.armed_stop(_pos(breakeven_px=98.0)) == 99.0


def test_observe_tracks_watermarks_immutably():
    p0 = _pos()
    p1 = manage.observe(p0, 100.4)
    p2 = manage.observe(p1, 99.6)
    p3 = manage.observe(p2, 101.2)

    assert (p0.high_px, p0.low_px) == (0.0, 0.0)
    assert (p3.high_px, p3.low_px) == (101.2, 99.6)


# --------------------------------------------------------------------------- #
# P&L / outcome
# --------------------------------------------------------------------------- #


def test_target_exit_pnl_is_net_of_both_fees():
    # sell 9.965 SOL @ 102, 0.35% fee in USDT
    proceeds = 9.965 * 102.0
    fee = proceeds * 0.0035
    pnl = manage.realized_pnl(_pos(), exit_px=102.0, exit_size=9.965, exit_fee_quote=fee)
    assert pnl == pytest.approx(proceeds - fee - 1_000.0)
    assert pnl == pytest.approx(12.87, abs=0.01)  # 2% gross -> ~1.29% net


def test_outcome_row_carries_r_multiple_and_excursions():
    pos = _pos(high_px=101.2, low_px=99.6)
    closed = T0 + timedelta(hours=2, minutes=30)
    row = manage.outcome(pos, exit_reason="stop", exit_px=99.0, exit_size=9.965, exit_fee_quote=3.45, closed_at=closed)

    expected_pnl = 99.0 * 9.965 - 3.45 - 1_000.0
    assert row["pnl_usd"] == pytest.approx(expected_pnl)
    assert row["r_multiple"] == pytest.approx(expected_pnl / 19.0)
    # a zero-slippage stop-out lands just inside -1R: the plan's risk budget
    # also reserves 0.2% round-trip slippage this fill didn't need
    assert -1.0 < row["r_multiple"] < -0.85
    assert row["mfe_frac"] == pytest.approx(0.012)
    assert row["mae_frac"] == pytest.approx(-0.004)
    assert row["time_in_trade_minutes"] == 150.0
    assert row["exit_reason"] == "stop"


def test_outcome_without_observations_reports_none_excursions():
    row = manage.outcome(_pos(), exit_reason="time", exit_px=100.0, exit_size=9.965, exit_fee_quote=0.0, closed_at=T0)
    assert row["mfe_frac"] is None and row["mae_frac"] is None


def test_positions_are_frozen():
    with pytest.raises(Exception):
        _pos().armed = True  # type: ignore[misc]
    assert replace(_pos(), armed=True).armed
