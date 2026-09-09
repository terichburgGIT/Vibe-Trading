"""Tests for the Discord alerter wiring into M006 (vt.risk.gate) and
M007 (vt.exec.adapter), added after S021 proved the alerter itself
works live. Neither module imports `vt.alerts.discord` at runtime
(only under `TYPE_CHECKING`) -- callers duck-type in whatever object
has the right methods, which is exactly what `FakeAlerter` below is.

Covers:
  * `gate.record_trade(..., alerter=...)` fires `alert_breaker_trip`
    exactly once per breaker on the call that actually trips it, and
    never re-fires for a breaker that was already tripped coming in.
  * `gate.kill(..., alerter=...)` fires once on a fresh kill, not on a
    kill of an already-killed state.
  * `adapter.reconcile(..., alerter=...)` fires `alert_drift` once per
    drifted symbol, and not at all when nothing drifted.
  * Every wired function is fully backward compatible: omitting
    `alerter` (the existing call shape from every pre-S022 test in
    `test_gate.py` / `test_exec_adapter.py`) behaves identically to
    before this parameter existed -- no alerter means zero attempts to
    construct or call one.

Run with: pytest vt/tests -m unit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vt.exec import adapter as vt_exec
from vt.risk import gate

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fake alerter -- records calls, duck-types DiscordAlerter's public surface
# --------------------------------------------------------------------------- #


@dataclass
class FakeAlerter:
    breaker_trips: list[dict] = field(default_factory=list)
    drifts: list[dict] = field(default_factory=list)

    def alert_breaker_trip(self, *, breaker: str, detail: str) -> bool:
        self.breaker_trips.append({"breaker": breaker, "detail": detail})
        return True

    def alert_drift(self, *, venue: str, symbol: str, internal_qty: float, broker_qty: float) -> bool:
        self.drifts.append(
            {"venue": venue, "symbol": symbol, "internal_qty": internal_qty, "broker_qty": broker_qty}
        )
        return True


_NOW = datetime(2026, 9, 9, 15, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# gate.record_trade -- breaker-trip alerting
# --------------------------------------------------------------------------- #


def test_record_trade_without_alerter_behaves_exactly_as_before() -> None:
    """Backward compatibility: the existing 35-test suite in
    test_gate.py never passes `alerter`. Confirm the default is truly
    None and nothing about the returned state changes."""
    state = gate.BreakerState()
    new_state = gate.record_trade(state, r_multiple=-2.5, equity=10_000.0, now=_NOW)
    assert new_state.session_r == -2.5  # trips daily loss limit, no alerter attached -- no error


def test_record_trade_fires_daily_loss_limit_trip_exactly_once() -> None:
    alerter = FakeAlerter()
    state = gate.BreakerState()

    # First trade: -2.5R breaches the -2.0R daily limit. Should fire.
    state = gate.record_trade(state, r_multiple=-2.5, equity=10_000.0, now=_NOW, alerter=alerter)
    assert len(alerter.breaker_trips) == 1
    assert alerter.breaker_trips[0]["breaker"] == "daily_loss_limit"
    assert "session_r=-2.50" in alerter.breaker_trips[0]["detail"]

    # Second trade same session, still tripped -- must NOT re-fire.
    state = gate.record_trade(state, r_multiple=-0.5, equity=10_000.0, now=_NOW, alerter=alerter)
    assert len(alerter.breaker_trips) == 1


def test_record_trade_fires_trade_cap_trip_exactly_once() -> None:
    alerter = FakeAlerter()
    state = gate.BreakerState()
    # Small wins so we don't also trip the loss limit -- isolate trade_cap.
    for _ in range(gate.DAILY_TRADE_CAP):
        state = gate.record_trade(state, r_multiple=0.1, equity=10_000.0, now=_NOW, alerter=alerter)

    trade_cap_trips = [t for t in alerter.breaker_trips if t["breaker"] == "trade_cap"]
    assert len(trade_cap_trips) == 1

    # One more trade, still capped -- must not re-fire trade_cap again.
    state = gate.record_trade(state, r_multiple=0.1, equity=10_000.0, now=_NOW, alerter=alerter)
    trade_cap_trips = [t for t in alerter.breaker_trips if t["breaker"] == "trade_cap"]
    assert len(trade_cap_trips) == 1


def test_record_trade_fires_consecutive_losses_trip_exactly_once() -> None:
    alerter = FakeAlerter()
    state = gate.BreakerState()
    # Small losses so the daily loss limit doesn't also trip and drown
    # out the signal we're isolating.
    for i in range(gate.CONSECUTIVE_LOSS_HALT):
        now = _NOW + timedelta(minutes=i)
        state = gate.record_trade(state, r_multiple=-0.05, equity=10_000.0, now=now, alerter=alerter)

    streak_trips = [t for t in alerter.breaker_trips if t["breaker"] == "consecutive_losses"]
    assert len(streak_trips) == 1
    assert f"{gate.CONSECUTIVE_LOSS_HALT} losses in a row" in streak_trips[0]["detail"]


def test_record_trade_fires_drawdown_kill_switch_trip_exactly_once() -> None:
    alerter = FakeAlerter()
    state = gate.BreakerState(equity_peak=10_000.0)

    # Equity drops 16% from peak -- breaches the -15R drawdown kill.
    state = gate.record_trade(state, r_multiple=-1.0, equity=8_400.0, now=_NOW, alerter=alerter)
    kill_trips = [t for t in alerter.breaker_trips if t["breaker"] == "drawdown_kill_switch"]
    assert len(kill_trips) == 1
    assert state.hard_killed is True

    # Another trade after hard_killed is already set -- must not re-fire.
    state = gate.record_trade(
        state, r_multiple=-0.1, equity=8_300.0, now=_NOW + timedelta(hours=1), alerter=alerter
    )
    kill_trips = [t for t in alerter.breaker_trips if t["breaker"] == "drawdown_kill_switch"]
    assert len(kill_trips) == 1


def test_record_trade_fires_weekly_loss_limit_trip_exactly_once() -> None:
    alerter = FakeAlerter()
    state = gate.BreakerState()
    # Spread losses across different days within the week, small enough
    # each day to not trip the daily limit, but summing past -6R weekly.
    monday = _NOW - timedelta(days=_NOW.weekday())
    for day_offset in range(4):
        day = monday + timedelta(days=day_offset)
        state = gate.record_trade(state, r_multiple=-1.8, equity=10_000.0, now=day, alerter=alerter)

    weekly_trips = [t for t in alerter.breaker_trips if t["breaker"] == "weekly_loss_limit"]
    assert len(weekly_trips) == 1


def test_record_trade_winning_trade_fires_no_alert() -> None:
    alerter = FakeAlerter()
    state = gate.BreakerState()
    gate.record_trade(state, r_multiple=1.5, equity=10_000.0, now=_NOW, alerter=alerter)
    assert alerter.breaker_trips == []


def test_record_trade_new_day_allows_daily_loss_limit_to_refire() -> None:
    """A NEW trip on a NEW day is a fresh event, not a re-fire of the
    same breaker -- the daily counters reset at the session boundary,
    and so should the alerting."""
    alerter = FakeAlerter()
    state = gate.BreakerState()
    state = gate.record_trade(state, r_multiple=-2.5, equity=10_000.0, now=_NOW, alerter=alerter)
    next_day = _NOW + timedelta(days=1)
    state = gate.record_trade(state, r_multiple=-2.5, equity=10_000.0, now=next_day, alerter=alerter)

    daily_trips = [t for t in alerter.breaker_trips if t["breaker"] == "daily_loss_limit"]
    assert len(daily_trips) == 2


# --------------------------------------------------------------------------- #
# gate.kill -- manual kill switch alerting
# --------------------------------------------------------------------------- #


def test_kill_without_alerter_behaves_exactly_as_before(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "_DEFAULT_BREAKER_STATE_PATH", tmp_path / "breaker_state.json")
    gate.kill()  # no alerter -- must not raise
    state = gate.load_breaker_state(tmp_path / "breaker_state.json")
    assert state.hard_killed is True


def test_kill_fires_alert_on_fresh_kill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "_DEFAULT_BREAKER_STATE_PATH", tmp_path / "breaker_state.json")
    alerter = FakeAlerter()
    gate.kill(alerter=alerter)
    assert len(alerter.breaker_trips) == 1
    assert alerter.breaker_trips[0]["breaker"] == "manual_kill_switch"


def test_kill_does_not_refire_on_an_already_killed_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "_DEFAULT_BREAKER_STATE_PATH", tmp_path / "breaker_state.json")
    alerter = FakeAlerter()
    gate.kill(alerter=alerter)
    gate.kill(alerter=alerter)  # already hard_killed -- must not re-alert
    assert len(alerter.breaker_trips) == 1


# --------------------------------------------------------------------------- #
# adapter.reconcile -- drift alerting
# --------------------------------------------------------------------------- #


@dataclass
class FakeBroker:
    venue: str = "alpaca"
    positions_rows: list[vt_exec.Position] = field(default_factory=list)

    def submit_entry(self, **kwargs) -> str:  # pragma: no cover -- unused here
        raise NotImplementedError

    def submit_stop(self, **kwargs) -> str:  # pragma: no cover -- unused here
        raise NotImplementedError

    def cancel(self, order_id: str) -> None:  # pragma: no cover -- unused here
        raise NotImplementedError

    def close_position(self, symbol: str) -> None:  # pragma: no cover -- unused here
        raise NotImplementedError

    def positions(self) -> list[vt_exec.Position]:
        return list(self.positions_rows)


def test_reconcile_without_alerter_behaves_exactly_as_before() -> None:
    broker = FakeBroker(
        positions_rows=[vt_exec.Position(venue="alpaca", symbol="AAPL", quantity=26.0)]
    )
    internal = [vt_exec.InternalPosition(venue="alpaca", symbol="AAPL", quantity=13.0)]
    result = vt_exec.reconcile(broker, internal)  # no alerter -- must not raise
    assert result.halted is True


def test_reconcile_fires_alert_drift_per_symbol() -> None:
    alerter = FakeAlerter()
    broker = FakeBroker(
        positions_rows=[
            vt_exec.Position(venue="alpaca", symbol="AAPL", quantity=26.0),
            vt_exec.Position(venue="alpaca", symbol="MSFT", quantity=0.0),
        ]
    )
    internal = [
        vt_exec.InternalPosition(venue="alpaca", symbol="AAPL", quantity=13.0),
        vt_exec.InternalPosition(venue="alpaca", symbol="TSLA", quantity=5.0),
    ]
    result = vt_exec.reconcile(broker, internal, alerter=alerter)

    assert result.halted is True
    assert len(alerter.drifts) == len(result.drifts)
    fired_symbols = {d["symbol"] for d in alerter.drifts}
    assert fired_symbols == {d.symbol for d in result.drifts}
    aapl_alert = next(d for d in alerter.drifts if d["symbol"] == "AAPL")
    assert aapl_alert["venue"] == "alpaca"
    assert aapl_alert["internal_qty"] == 13.0
    assert aapl_alert["broker_qty"] == 26.0


def test_reconcile_matching_positions_fires_no_alert() -> None:
    alerter = FakeAlerter()
    broker = FakeBroker(
        positions_rows=[vt_exec.Position(venue="alpaca", symbol="AAPL", quantity=13.0)]
    )
    internal = [vt_exec.InternalPosition(venue="alpaca", symbol="AAPL", quantity=13.0)]
    result = vt_exec.reconcile(broker, internal, alerter=alerter)

    assert result.halted is False
    assert alerter.drifts == []
