"""Tests for the live `GateState` feeder (`vt.pipeline.gate_feeder`).

Two layers:
  * Pure vol helpers (`_daily_returns`, `_rolling_stdev`,
    `_realized_vol_inputs`) -- hand-verified.
  * `build_gate_state` over a fake feed -- proves it derives the M004
    numeric inputs from index daily bars and resolves the right state
    (NORMAL / REDUCED / STAND_DOWN), including the injected-VIX path and
    the fail-loud-on-no-bars path.

The state-resolution *logic* (precedence, percentile threshold, 2-sigma
check) is M004's contract, tested in `test_calendar_gate.py`; here we only
assert the feeder produces the right inputs and forwards them.

Run with: pytest vt/tests -m unit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vt.data.feed import Bar
from vt.gate.calendar import GateState
from vt.pipeline.gate_feeder import (
    GateFeederError,
    _daily_returns,
    _realized_vol_inputs,
    _rolling_stdev,
    build_gate_state,
)

pytestmark = pytest.mark.unit


ASOF = datetime(2026, 9, 9, 20, 0, 0, tzinfo=timezone.utc)


def _daily_bar(day: datetime, close: float, *, symbol="SPY", source_feed="alpaca_iex") -> Bar:
    return Bar(time=day, open=close, high=close, low=close, close=close, volume=1e6,
               symbol=symbol, source_feed=source_feed)


def _daily_series(closes: list[float], *, symbol="SPY", source_feed="alpaca_iex",
                  end: datetime = ASOF) -> list[Bar]:
    """Daily bars, one per day, ending at `end` (most recent last)."""
    n = len(closes)
    return [
        _daily_bar(end - timedelta(days=(n - 1 - i)), c, symbol=symbol, source_feed=source_feed)
        for i, c in enumerate(closes)
    ]


@dataclass
class FakeFeed:
    bars_by_key: dict[tuple[str, str], list[Bar]] = field(default_factory=dict)

    def get_bars(self, symbol, timeframe="1d", *, start=None, end=None, limit=90):  # noqa: ARG002
        return list(self.bars_by_key.get((symbol, timeframe), []))

    def get_quote(self, symbol):  # pragma: no cover - feeder never quotes
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_daily_returns_are_fractional_close_to_close() -> None:
    bars = _daily_series([100.0, 101.0, 99.0])
    rets = _daily_returns(bars)
    assert rets == pytest.approx([0.01, -0.019801980198])


def test_daily_returns_skips_nonpositive_prior_close() -> None:
    bars = _daily_series([0.0, 100.0, 110.0])
    # first ratio (against 0) is skipped; only 100->110 survives.
    assert _daily_returns(bars) == pytest.approx([0.1])


def test_rolling_stdev_window_and_alignment() -> None:
    returns = [0.01, -0.01, 0.02, -0.02]
    vols = _rolling_stdev(returns, 2)
    assert len(vols) == 3  # windows: [0,1], [1,2], [2,3]
    # last window stdev of [0.02, -0.02] = sample stdev
    import statistics
    assert vols[-1] == pytest.approx(statistics.stdev([0.02, -0.02]))


def test_rolling_stdev_empty_when_too_few_returns() -> None:
    assert _rolling_stdev([0.01], 2) == []
    assert _rolling_stdev([0.01, 0.02], 5) == []


def test_realized_vol_inputs_excludes_today_from_history() -> None:
    returns = [0.01, -0.01, 0.02, -0.02, 0.03, -0.03]
    rv, hist = _realized_vol_inputs(returns, window=2, history_len=60)
    vols = _rolling_stdev(returns, 2)
    assert rv == vols[-1]
    assert hist == vols[:-1]  # today excluded


def test_realized_vol_inputs_none_when_no_full_window() -> None:
    rv, hist = _realized_vol_inputs([0.01], window=2, history_len=60)
    assert rv is None and hist is None


def test_realized_vol_history_capped_at_history_len() -> None:
    returns = [0.01 * (1 if i % 2 else -1) for i in range(50)]
    _, hist = _realized_vol_inputs(returns, window=2, history_len=5)
    assert hist is not None
    assert len(hist) == 5


# --------------------------------------------------------------------------- #
# build_gate_state
# --------------------------------------------------------------------------- #


def _healthy_index() -> list[Bar]:
    """A daily index series with ordinary (~1%) day-to-day moves so realized
    vol is neither dead-tape-low nor a big-move outlier."""
    closes = [100.0]
    for i in range(90):
        closes.append(closes[-1] * (1.0 + (0.01 if i % 2 else -0.008)))
    return _daily_series(closes)


def test_normal_state_with_healthy_index_and_low_vix() -> None:
    feed = FakeFeed(bars_by_key={("SPY", "1d"): _healthy_index()})
    state = build_gate_state(feed, ASOF, vix=15.0)
    assert isinstance(state, GateState)
    assert state.state == "NORMAL"
    assert state.multiplier == 1.0


def test_injected_high_vix_reduces() -> None:
    feed = FakeFeed(bars_by_key={("SPY", "1d"): _healthy_index()})
    state = build_gate_state(feed, ASOF, vix=35.0)
    assert state.state == "REDUCED"
    assert state.multiplier == 0.5
    assert any("vix" in r for r in state.reasons)


def test_dead_tape_low_realized_vol_stands_down() -> None:
    """Older history swings ~2%/day; the most recent window goes nearly
    flat. Today's realized vol should fall below the 20th percentile of the
    trailing history -> STAND_DOWN."""
    closes = [100.0]
    for i in range(70):  # volatile history
        closes.append(closes[-1] * (1.0 + (0.02 if i % 2 else -0.02)))
    for _ in range(10):  # recent dead tape
        closes.append(closes[-1] * 1.0001)
    feed = FakeFeed(bars_by_key={("SPY", "1d"): _daily_series(closes)})

    state = build_gate_state(feed, ASOF, vix=15.0, realized_vol_window=5)

    assert state.state == "STAND_DOWN"
    assert state.multiplier == 0.0
    assert any("realized_vol" in r for r in state.reasons)


def test_crypto_venue_uses_btc_index() -> None:
    btc = _daily_series(
        [100.0 * (1.0 + 0.01 * (1 if i % 2 else -1)) for i in range(60)],
        symbol="BTC-USDT", source_feed="okx_demo",
    )
    feed = FakeFeed(bars_by_key={("BTC-USDT", "1d"): btc})
    # No SPY in the feed at all -- if it fetched SPY this would raise.
    state = build_gate_state(feed, ASOF, venue="okx", vix=None)
    assert isinstance(state, GateState)
    assert state.state in {"NORMAL", "REDUCED", "STAND_DOWN"}


def test_explicit_index_symbol_override() -> None:
    qqq = _daily_series(
        [100.0 * (1.0 + 0.01 * (1 if i % 2 else -1)) for i in range(60)], symbol="QQQ"
    )
    feed = FakeFeed(bars_by_key={("QQQ", "1d"): qqq})
    state = build_gate_state(feed, ASOF, index_symbol="QQQ", vix=12.0)
    assert isinstance(state, GateState)


def test_raises_when_no_index_bars() -> None:
    feed = FakeFeed(bars_by_key={})  # nothing for SPY
    with pytest.raises(GateFeederError, match="no daily bars"):
        build_gate_state(feed, ASOF, vix=15.0)


def test_short_history_degrades_but_does_not_raise() -> None:
    """Only a handful of daily bars: not enough for a full realized-vol
    window, so the vol trigger is skipped -- but the feeder still returns a
    valid GateState (NORMAL here) rather than raising."""
    feed = FakeFeed(bars_by_key={("SPY", "1d"): _daily_series([100.0, 101.0, 100.5])})
    state = build_gate_state(feed, ASOF, vix=15.0, realized_vol_window=10)
    assert isinstance(state, GateState)
    assert state.state == "NORMAL"


def test_event_day_stands_down_via_yaml(tmp_path: Path) -> None:
    """A macro-print date in the events YAML forces STAND_DOWN even with a
    perfectly healthy index -- proves the feeder forwards events_path."""
    events = tmp_path / "events.yaml"
    events.write_text(
        f"stand_down:\n  - date: {ASOF.date().isoformat()}\n    event: FOMC\n",
        encoding="utf-8",
    )
    feed = FakeFeed(bars_by_key={("SPY", "1d"): _healthy_index()})
    state = build_gate_state(feed, ASOF, vix=15.0, events_path=events)
    assert state.state == "STAND_DOWN"
    assert any("FOMC" in r for r in state.reasons)
