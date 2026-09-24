"""Tests for M010 -- vt.validate.walk_forward (T020; see 06_Tests.md).

Run with: pytest vt/tests -m unit
"""

from __future__ import annotations

import dataclasses
import math
from datetime import datetime, timedelta, timezone

import pytest

from vt.data.feed import Bar
from vt.indicators.engine import IndicatorFrame, compute
from vt.validate import walk_forward as wf

pytestmark = pytest.mark.unit


def _bar(dt: datetime, *, close: float, volume: float = 1_000.0, symbol: str = "AAPL", source_feed: str = "alpaca_iex") -> Bar:
    return Bar(time=dt, open=close, high=close + 0.5, low=close - 0.5, close=close, volume=volume, symbol=symbol, source_feed=source_feed)


def _series(closes: list[float], *, start: datetime | None = None) -> list[Bar]:
    """A trending, non-degenerate close series -- enough bars past every
    indicator's warmup (ADX14/ATR14/RSI14 all need >= 14) for real
    (non-None) values to actually exist at the tail end, which is where
    lookahead is easiest to hide."""
    start = start or datetime(2026, 9, 1, tzinfo=timezone.utc)
    return [_bar(start + timedelta(minutes=i), close=c) for i, c in enumerate(closes)]


def _trending_bars(n: int = 30) -> list[Bar]:
    # Not perfectly linear -- a flat/linear series degenerates ADX's DM
    # calculations. Small wobble keeps every indicator genuinely live.
    closes = [100.0 + i * 0.7 + (0.3 if i % 3 == 0 else -0.1) for i in range(n)]
    return _series(closes)


# --------------------------------------------------------------------------- #
# bars_with_future_nan
# --------------------------------------------------------------------------- #


def test_bars_with_future_nan_leaves_past_and_current_bars_untouched() -> None:
    bars = _trending_bars(10)
    masked = wf.bars_with_future_nan(bars, as_of_index=4)

    for i in range(5):
        assert masked[i] == bars[i]


def test_bars_with_future_nan_nans_ohlcv_but_keeps_time_symbol_source_feed() -> None:
    bars = _trending_bars(10)
    masked = wf.bars_with_future_nan(bars, as_of_index=4)

    for i in range(5, 10):
        assert math.isnan(masked[i].open)
        assert math.isnan(masked[i].high)
        assert math.isnan(masked[i].low)
        assert math.isnan(masked[i].close)
        assert math.isnan(masked[i].volume)
        assert masked[i].time == bars[i].time
        assert masked[i].symbol == bars[i].symbol
        assert masked[i].source_feed == bars[i].source_feed


def test_bars_with_future_nan_rejects_out_of_range_index() -> None:
    bars = _trending_bars(5)
    with pytest.raises(ValueError):
        wf.bars_with_future_nan(bars, as_of_index=5)
    with pytest.raises(ValueError):
        wf.bars_with_future_nan(bars, as_of_index=-1)


# --------------------------------------------------------------------------- #
# T020 -- the real M003 compute() has no lookahead
# --------------------------------------------------------------------------- #


def test_real_indicator_engine_has_no_lookahead_violations() -> None:
    """The headline assertion: M003's actual compute() function, run against
    a real trending bar series long enough for every indicator to be past
    warmup, produces zero lookahead violations."""
    bars = _trending_bars(30)
    violations = wf.find_lookahead_violations(bars)
    assert violations == []


def test_assert_no_lookahead_does_not_raise_on_the_real_engine() -> None:
    bars = _trending_bars(30)
    wf.assert_no_lookahead(bars)  # must not raise


def test_no_lookahead_holds_on_a_short_series_still_in_warmup() -> None:
    """Fewer bars than any indicator's warmup period -- every value is None,
    but that's not lookahead, it's honest 'not enough data yet' (per
    IndicatorFrame's own docstring). Must still report zero violations."""
    bars = _trending_bars(5)
    assert wf.find_lookahead_violations(bars) == []


def test_no_lookahead_holds_on_a_single_bar() -> None:
    bars = _trending_bars(1)
    assert wf.find_lookahead_violations(bars) == []


def test_no_lookahead_on_empty_bars_is_trivially_clean() -> None:
    assert wf.find_lookahead_violations([]) == []


# --------------------------------------------------------------------------- #
# T020 -- the detector actually catches a real lookahead bug (mirror test)
# --------------------------------------------------------------------------- #


def _compute_with_injected_lookahead(bars):
    """A deliberately broken compute_fn: ema9 at index i is corrupted to
    equal the (real, lookahead-free) ema9 that would only be known at
    index i+1 -- i.e. it reads one bar into the future. This function
    exists ONLY to prove find_lookahead_violations actually detects a
    known-bad implementation, not just agrees with an already-correct
    one (same 'mirror test' shape as T011/T027 elsewhere in this repo)."""
    frame = compute(bars)
    if len(frame.ema9) < 2:
        return frame
    shifted_ema9 = frame.ema9[1:] + [frame.ema9[-1]]
    return dataclasses.replace(frame, ema9=shifted_ema9)


def test_detector_catches_a_deliberately_injected_lookahead_bug() -> None:
    """The mirror of the headline test: feed the SAME real bars through a
    compute_fn with a known, deliberate lookahead defect, and confirm
    violations are reported -- proving the detector has teeth, not just
    a rubber stamp."""
    bars = _trending_bars(30)
    violations = wf.find_lookahead_violations(bars, compute_fn=_compute_with_injected_lookahead)

    assert len(violations) > 0
    assert all(v.field == "ema9" for v in violations)


def test_assert_no_lookahead_raises_on_the_injected_bug() -> None:
    bars = _trending_bars(30)
    with pytest.raises(wf.LookaheadDetectedError):
        wf.assert_no_lookahead(bars, compute_fn=_compute_with_injected_lookahead)


def test_lookahead_error_message_names_the_first_violation() -> None:
    bars = _trending_bars(30)
    with pytest.raises(wf.LookaheadDetectedError) as exc_info:
        wf.assert_no_lookahead(bars, compute_fn=_compute_with_injected_lookahead)

    message = str(exc_info.value)
    assert "ema9" in message
    assert "bar_index=" in message
