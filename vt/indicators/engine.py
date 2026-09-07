"""M003 -- Indicator Engine.

Computes the six rubric inputs (VWAP, EMA9/21, RSI14, OBV, ADX14, ATR14)
for a bar series. Net-new, hand-rolled -- `pandas-ta` was scoped in the
original module notes but isn't an installed dependency and has had
maintenance gaps; six well-documented formulas don't need a third-party
wrapper, and hand-rolling keeps every step exact and golden-value
testable to the precision T005 asks for (see `vt/tests/test_engine.py`).

Every function here is a **pure, forward-only** recurrence: computing
index i only ever reads bars[0..i], never bars[i+1:]. This is a design
invariant, not an accident -- M010's walk-forward validation (T020, "no
indicator at bar t uses data from t+1") depends on it holding from day
one, so nothing here is written to look ahead even provisionally.

VWAP resets at the equity session boundary (a change in `bar.time.date()`
-- UTC calendar day, consistent with M001/M002's UTC-internal convention;
this is a simplification that doesn't know exchange session hours or
holidays, which is M004's job, not this module's) and uses a rolling 24h
window for crypto (venue inferred from `bar.source_feed`, never a
separate caller-supplied flag that could drift out of sync with the
actual bars).

RSI, ATR, and ADX all use Wilder's smoothing method (seed = simple
average of the first `period` values, then
`smoothed_t = (smoothed_{t-1} * (period-1) + value_t) / period`) --
the standard, textbook form for all three, applied consistently.

`vt.universe.screen._atr14_pct` has its own local, simpler ATR (a plain
moving average of true ranges, not Wilder-smoothed) -- deliberately not
shared with this module yet, per that module's own docstring ("if the
two ever diverge, reconcile then"). This module's `atr()` is the
Wilder-correct one and is the one T005 tests to 4dp.

Full contract: `03_Modules.md` § M003. Tests: `vt/tests/test_engine.py`
(T005 in `06_Tests.md`).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence

from vt.data.feed import Bar

_ROLLING_VWAP_WINDOW = timedelta(hours=24)


@dataclass(frozen=True)
class IndicatorFrame:
    """Parallel arrays aligned 1:1 with the input `bars` -- index i is bar i.

    `None` at a given index means insufficient warmup data at that point
    (e.g. the first 13 entries of `rsi14`), never a silently-wrong zero.
    `vwap` and `obv` have no warmup period, so their entries are never
    `None` for a non-empty input (`vwap` can still be `None` at an index
    with zero cumulative window volume -- see `vwap()`).
    """

    vwap: list[float | None]
    ema9: list[float | None]
    ema21: list[float | None]
    rsi14: list[float | None]
    obv: list[float]
    adx14: list[float | None]
    atr14: list[float | None]


def _is_crypto(bars: Sequence[Bar]) -> bool:
    return bool(bars) and bars[0].source_feed.startswith("okx")


def _typical_price(bar: Bar) -> float:
    return (bar.high + bar.low + bar.close) / 3.0


def vwap(bars: Sequence[Bar]) -> list[float | None]:
    """Volume-weighted average price, aligned to `bars`.

    Equities reset at each session boundary (a change in `bar.time.date()`).
    Crypto uses a rolling 24h window instead -- no session concept, since
    it trades continuously. Venue is inferred from `bars[0].source_feed`
    ("okx*" -> crypto, everything else -> equity); a mixed-venue `bars`
    list isn't a case this module needs to handle -- callers always pull
    one symbol's bars from one venue via `vt.data.feed`.

    Returns `None` at any index where the window's cumulative volume is
    zero (a divide-by-zero would otherwise be silently wrong).
    """
    if not bars:
        return []

    result: list[float | None] = []
    if _is_crypto(bars):
        window: deque[Bar] = deque()
        cum_pv = 0.0
        cum_vol = 0.0
        for bar in bars:
            window.append(bar)
            cum_pv += _typical_price(bar) * bar.volume
            cum_vol += bar.volume
            cutoff = bar.time - _ROLLING_VWAP_WINDOW
            while window and window[0].time < cutoff:
                stale = window.popleft()
                cum_pv -= _typical_price(stale) * stale.volume
                cum_vol -= stale.volume
            result.append(cum_pv / cum_vol if cum_vol > 0 else None)
    else:
        cum_pv = 0.0
        cum_vol = 0.0
        current_session: date | None = None
        for bar in bars:
            if bar.time.date() != current_session:
                current_session = bar.time.date()
                cum_pv = 0.0
                cum_vol = 0.0
            cum_pv += _typical_price(bar) * bar.volume
            cum_vol += bar.volume
            result.append(cum_pv / cum_vol if cum_vol > 0 else None)
    return result


def ema(values: Sequence[float], period: int) -> list[float | None]:
    """Exponential moving average, generic over `period`.

    First `period - 1` entries are `None` (insufficient warmup). Index
    `period - 1` seeds from the simple average of the first `period`
    values; every entry after follows the standard recurrence
    `EMA_t = value_t * k + EMA_{t-1} * (1 - k)`, `k = 2 / (period + 1)`.
    """
    n = len(values)
    result: list[float | None] = [None] * n
    if n < period:
        return result
    k = 2.0 / (period + 1)
    seed = sum(values[:period]) / period
    result[period - 1] = seed
    prev = seed
    for i in range(period, n):
        prev = values[i] * k + prev * (1 - k)
        result[i] = prev
    return result


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def rsi(closes: Sequence[float], period: int = 14) -> list[float | None]:
    """Wilder's RSI over `closes`.

    First `period` entries are `None` (need `period` deltas, i.e.
    `period + 1` closes, before the first value at index `period`).
    `avg_loss == 0` (every delta so far a gain) reads as RSI == 100
    exactly, rather than dividing by zero.
    """
    n = len(closes)
    result: list[float | None] = [None] * n
    if n <= period:
        return result

    gains = [max(closes[i] - closes[i - 1], 0.0) for i in range(1, n)]
    losses = [max(closes[i - 1] - closes[i], 0.0) for i in range(1, n)]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    result[period] = _rsi_from_averages(avg_gain, avg_loss)

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        result[i + 1] = _rsi_from_averages(avg_gain, avg_loss)
    return result


def obv(bars: Sequence[Bar]) -> list[float]:
    """On-Balance Volume. `obv[0]` is 0 (no prior close to compare against);
    each subsequent bar adds its volume on a higher close, subtracts it on
    a lower close, and carries forward unchanged on an equal close.
    """
    n = len(bars)
    result: list[float] = [0.0] * n
    for i in range(1, n):
        if bars[i].close > bars[i - 1].close:
            result[i] = result[i - 1] + bars[i].volume
        elif bars[i].close < bars[i - 1].close:
            result[i] = result[i - 1] - bars[i].volume
        else:
            result[i] = result[i - 1]
    return result


def _true_range(bar: Bar, prev_close: float) -> float:
    return max(bar.high - bar.low, abs(bar.high - prev_close), abs(bar.low - prev_close))


def atr(bars: Sequence[Bar], period: int = 14) -> list[float | None]:
    """Wilder's ATR over `bars`. First `period` entries are `None`."""
    n = len(bars)
    result: list[float | None] = [None] * n
    if n <= period:
        return result

    trs = [_true_range(bars[i], bars[i - 1].close) for i in range(1, n)]
    atr_val = sum(trs[:period]) / period
    result[period] = atr_val
    for i in range(period, len(trs)):
        atr_val = (atr_val * (period - 1) + trs[i]) / period
        result[i + 1] = atr_val
    return result


def _wilder_di_and_dx(smoothed_tr: float, smoothed_plus_dm: float, smoothed_minus_dm: float) -> float:
    if smoothed_tr <= 0:
        return 0.0
    plus_di = 100.0 * smoothed_plus_dm / smoothed_tr
    minus_di = 100.0 * smoothed_minus_dm / smoothed_tr
    denom = plus_di + minus_di
    return 0.0 if denom == 0 else 100.0 * abs(plus_di - minus_di) / denom


def adx(bars: Sequence[Bar], period: int = 14) -> list[float | None]:
    """Wilder's ADX over `bars`.

    Needs `period` bars to seed smoothed TR/+DM/-DM, then another
    `period` DX readings to seed ADX itself -- the first non-`None` value
    lands at index `2 * period - 1`.
    """
    n = len(bars)
    result: list[float | None] = [None] * n
    if n < 2 * period:
        return result

    plus_dm = [0.0] * n
    minus_dm = [0.0] * n
    tr = [0.0] * n
    for i in range(1, n):
        up_move = bars[i].high - bars[i - 1].high
        down_move = bars[i - 1].low - bars[i].low
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
        tr[i] = _true_range(bars[i], bars[i - 1].close)

    smoothed_tr = sum(tr[1 : period + 1]) / period
    smoothed_plus_dm = sum(plus_dm[1 : period + 1]) / period
    smoothed_minus_dm = sum(minus_dm[1 : period + 1]) / period

    dx_values = [_wilder_di_and_dx(smoothed_tr, smoothed_plus_dm, smoothed_minus_dm)]
    for i in range(period + 1, n):
        smoothed_tr = (smoothed_tr * (period - 1) + tr[i]) / period
        smoothed_plus_dm = (smoothed_plus_dm * (period - 1) + plus_dm[i]) / period
        smoothed_minus_dm = (smoothed_minus_dm * (period - 1) + minus_dm[i]) / period
        dx_values.append(_wilder_di_and_dx(smoothed_tr, smoothed_plus_dm, smoothed_minus_dm))
    # dx_values[k] corresponds to bar index (period + k).

    if len(dx_values) < period:
        return result

    adx_val = sum(dx_values[:period]) / period
    result[2 * period - 1] = adx_val
    for k in range(period, len(dx_values)):
        adx_val = (adx_val * (period - 1) + dx_values[k]) / period
        result[period + k] = adx_val
    return result


def compute(bars: Sequence[Bar]) -> IndicatorFrame:
    """Compute all six rubric indicators (+ VWAP), aligned to `bars`.

    Pure aggregation over the module's other functions -- see the module
    docstring for the no-lookahead guarantee this (and every function it
    calls) maintains.
    """
    closes = [b.close for b in bars]
    return IndicatorFrame(
        vwap=vwap(bars),
        ema9=ema(closes, 9),
        ema21=ema(closes, 21),
        rsi14=rsi(closes, 14),
        obv=obv(bars),
        adx14=adx(bars, 14),
        atr14=atr(bars, 14),
    )
