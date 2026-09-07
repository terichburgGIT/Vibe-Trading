"""Tests for M003 -- vt.indicators.engine (T005; see 06_Tests.md).

Unit only, all synthetic, all hand-computed -- no network. Every expected
value here is derived by hand in the test itself (spelled out in comments),
never by running the implementation and pasting its output back as the
"expected" value -- that would be circular and prove nothing.

Run with: pytest vt/tests -m unit
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from vt.data.feed import Bar
from vt.indicators import engine

pytestmark = pytest.mark.unit


def _bar(
    dt: datetime,
    *,
    high: float,
    low: float,
    close: float,
    volume: float,
    symbol: str = "AAPL",
    source_feed: str = "alpaca_iex",
) -> Bar:
    return Bar(time=dt, open=close, high=high, low=low, close=close, volume=volume, symbol=symbol, source_feed=source_feed)


def _flat_bar(dt: datetime, close: float, volume: float, **kw: object) -> Bar:
    return _bar(dt, high=close, low=close, close=close, volume=volume, **kw)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# ema() -- generic-period EMA, tested at period=3 for tractable arithmetic
# (ema9/ema21 in compute() are the same function at different periods)
# --------------------------------------------------------------------------- #


def test_ema_matches_hand_computed_values() -> None:
    """closes = [10, 11, 12, 13, 14], period=3, k = 2/(3+1) = 0.5.

    seed (index 2) = avg(10, 11, 12) = 11.0
    index 3: 13*0.5 + 11.0*0.5 = 12.0
    index 4: 14*0.5 + 12.0*0.5 = 13.0
    """
    closes = [10.0, 11.0, 12.0, 13.0, 14.0]
    result = engine.ema(closes, 3)
    assert result[0] is None
    assert result[1] is None
    assert result[2] == pytest.approx(11.0, abs=1e-4)
    assert result[3] == pytest.approx(12.0, abs=1e-4)
    assert result[4] == pytest.approx(13.0, abs=1e-4)


def test_ema_all_none_when_fewer_bars_than_period() -> None:
    assert engine.ema([1.0, 2.0], 5) == [None, None]


# --------------------------------------------------------------------------- #
# rsi() -- Wilder's RSI(14)
# --------------------------------------------------------------------------- #


def test_rsi_matches_hand_computed_mixed_series() -> None:
    """closes = [10,11,10,11,10,11,10,11,10,11,10,11,10,11,12] (15 values, 14 deltas).

    deltas: +1,-1,+1,-1,+1,-1,+1,-1,+1,-1,+1,-1,+1,+1
    gains:  1,0,1,0,1,0,1,0,1,0,1,0,1,1 -> sum = 8, avg_gain = 8/14
    losses: 0,1,0,1,0,1,0,1,0,1,0,1,0,0 -> sum = 6, avg_loss = 6/14
    RS = (8/14) / (6/14) = 8/6 = 4/3
    RSI = 100 - 100/(1 + 4/3) = 100 - 300/7 = 400/7 = 57.142857...
    """
    closes = [10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 12]
    result = engine.rsi([float(c) for c in closes], 14)
    assert result[13] is None  # index 14 (the 15th close) is the first computable RSI value
    assert result[14] == pytest.approx(400.0 / 7.0, abs=1e-4)


def test_rsi_is_100_when_every_delta_is_a_gain() -> None:
    """Strictly increasing closes -> avg_loss stays exactly 0 -> RSI == 100
    at every computable index (exercises the avg_loss == 0 edge branch).
    """
    closes = [float(10 + i) for i in range(16)]  # 16 closes, all deltas = +1
    result = engine.rsi(closes, 14)
    assert result[14] == pytest.approx(100.0, abs=1e-9)
    assert result[15] == pytest.approx(100.0, abs=1e-9)


def test_rsi_none_before_warmup() -> None:
    closes = [float(i) for i in range(10)]
    result = engine.rsi(closes, 14)
    assert all(v is None for v in result)


# --------------------------------------------------------------------------- #
# obv() -- On-Balance Volume
# --------------------------------------------------------------------------- #


def test_obv_matches_hand_computed_values() -> None:
    """closes=[10,11,11,9,10], volumes=[100,200,150,300,120].

    OBV[0] = 0 (no prior close)
    OBV[1] = 0 + 200   (11 > 10, up)          = 200
    OBV[2] = 200       (11 == 11, flat)       = 200
    OBV[3] = 200 - 300 (9 < 11, down)         = -100
    OBV[4] = -100 + 120 (10 > 9, up)          = 20
    """
    base = datetime(2026, 9, 1, tzinfo=timezone.utc)
    closes = [10.0, 11.0, 11.0, 9.0, 10.0]
    volumes = [100.0, 200.0, 150.0, 300.0, 120.0]
    bars = [_flat_bar(base + timedelta(minutes=i), c, v) for i, (c, v) in enumerate(zip(closes, volumes))]

    result = engine.obv(bars)

    assert result == pytest.approx([0.0, 200.0, 200.0, -100.0, 20.0])


# --------------------------------------------------------------------------- #
# atr() -- Wilder's ATR(14)
# --------------------------------------------------------------------------- #


def test_atr14_matches_hand_computed_seed_and_smoothed_update() -> None:
    """Seed bar (index 0) close=101. Bars 1-14: low=100, high=102, close=101
    each -> TR = high-low = 2 (gap terms |102-101|=1, |100-101|=1, both < 2).
    Bar 15: low=90, high=106, close=98 -> TR = high-low = 16 (gap terms
    |106-101|=5, |90-101|=11, both < 16, so range term still dominates).

    seed (index 14) = avg(fourteen TRs of 2) = 2.0
    index 15 (Wilder update) = (2.0*13 + 16) / 14 = 42/14 = 3.0
    """
    base = datetime(2026, 9, 1, tzinfo=timezone.utc)
    bars = [_bar(base, high=101, low=101, close=101, volume=1)]
    for i in range(1, 15):
        bars.append(_bar(base + timedelta(minutes=i), high=102, low=100, close=101, volume=1))
    bars.append(_bar(base + timedelta(minutes=15), high=106, low=90, close=98, volume=1))

    result = engine.atr(bars, 14)

    assert all(v is None for v in result[:14])
    assert result[14] == pytest.approx(2.0, abs=1e-9)
    assert result[15] == pytest.approx(3.0, abs=1e-9)


def test_atr14_none_before_warmup() -> None:
    base = datetime(2026, 9, 1, tzinfo=timezone.utc)
    bars = [_bar(base + timedelta(minutes=i), high=11, low=9, close=10, volume=1) for i in range(5)]
    assert all(v is None for v in engine.atr(bars, 14))


# --------------------------------------------------------------------------- #
# adx() -- Wilder's ADX(14). Two degenerate-but-exact constructions:
# a clean uptrend (-DM always 0 -> DX == 100 exactly) and a flat series
# (TR == 0 -> DX == 0 exactly). Together they prove the pipeline isn't
# just returning a hardcoded constant.
# --------------------------------------------------------------------------- #


def test_adx14_is_exactly_100_on_a_clean_monotonic_uptrend() -> None:
    """30 bars, each low/high/close shifted +2 from the previous bar
    (low=100+2i, high=102+2i, close=101+2i). Every bar: up_move=+2,
    down_move=-2 (clamped to 0) -> minus_dm=0 for every bar, forever.
    Whenever minus_dm==0, minus_di==0, so DX = 100*(plus_di-0)/(plus_di+0)
    == 100.0 exactly, regardless of plus_di's actual value -- and since
    minus_dm never recovers from 0, DX stays 100.0 at every computable
    index, so the Wilder-averaged ADX is also 100.0 throughout.
    """
    base = datetime(2026, 9, 1, tzinfo=timezone.utc)
    bars = [_bar(base + timedelta(minutes=i), high=102 + 2 * i, low=100 + 2 * i, close=101 + 2 * i, volume=1) for i in range(30)]

    result = engine.adx(bars, 14)

    assert all(v is None for v in result[:27])  # needs 2*period-1 = 27 bars of warmup
    assert result[27] == pytest.approx(100.0, abs=1e-6)
    assert result[28] == pytest.approx(100.0, abs=1e-6)
    assert result[29] == pytest.approx(100.0, abs=1e-6)


def test_adx14_is_exactly_zero_on_a_flat_series() -> None:
    """No price movement at all -> TR == 0 for every bar -> the zero-TR
    guard in DI/DX returns DX == 0.0 exactly, so ADX averages to 0.0.
    """
    base = datetime(2026, 9, 1, tzinfo=timezone.utc)
    bars = [_flat_bar(base + timedelta(minutes=i), 100.0, 1.0) for i in range(30)]

    result = engine.adx(bars, 14)

    assert result[27] == pytest.approx(0.0, abs=1e-9)
    assert result[29] == pytest.approx(0.0, abs=1e-9)


def test_adx14_none_before_warmup() -> None:
    base = datetime(2026, 9, 1, tzinfo=timezone.utc)
    bars = [_bar(base + timedelta(minutes=i), high=102 + i, low=100 + i, close=101 + i, volume=1) for i in range(20)]
    assert all(v is None for v in engine.adx(bars, 14))


# --------------------------------------------------------------------------- #
# vwap() -- session-reset for equities, rolling 24h for crypto
# --------------------------------------------------------------------------- #


def test_vwap_resets_at_equity_session_boundary() -> None:
    """Two sessions, two bars each. Typical price == close here (flat
    high=low=close), so pv = close * volume.

    Session 1 (day 1): bar1 (close=10, vol=100) -> pv=1000
                        bar2 (close=20, vol=100) -> pv=2000
      VWAP1 = 1000/100 = 10.0
      VWAP2 = (1000+2000)/200 = 15.0
    Session 2 (day 2, RESET -- must not carry day 1's 3000/200 forward):
                        bar3 (close=5,  vol=50)  -> pv=250
                        bar4 (close=15, vol=50)  -> pv=750
      VWAP3 = 250/50 = 5.0   (if it wrongly carried over: (3000+250)/250=13.0)
      VWAP4 = (250+750)/100 = 10.0
    """
    day1 = datetime(2026, 9, 1, 14, 0, tzinfo=timezone.utc)
    day2 = datetime(2026, 9, 2, 14, 0, tzinfo=timezone.utc)
    bars = [
        _flat_bar(day1, 10.0, 100.0, source_feed="alpaca_iex"),
        _flat_bar(day1 + timedelta(minutes=1), 20.0, 100.0, source_feed="alpaca_iex"),
        _flat_bar(day2, 5.0, 50.0, source_feed="alpaca_iex"),
        _flat_bar(day2 + timedelta(minutes=1), 15.0, 50.0, source_feed="alpaca_iex"),
    ]

    result = engine.vwap(bars)

    assert result == pytest.approx([10.0, 15.0, 5.0, 10.0])


def test_vwap_uses_rolling_24h_window_for_crypto() -> None:
    """bar_a: Day1 00:00, close=100, vol=1 -> pv=100
    bar_b: Day1 12:00, close=200, vol=1 -> pv=200
    bar_c: Day2 01:00, close=300, vol=1 -> pv=300 (25h after bar_a)

    cutoff for bar_c = bar_c.time - 24h = Day1 01:00.
    bar_a (Day1 00:00) < cutoff -> evicted. bar_b (Day1 12:00) >= cutoff -> stays.

    VWAP_a = 100/1 = 100.0
    VWAP_b = (100+200)/2 = 150.0
    VWAP_c (after evicting a) = (200+300)/2 = 250.0   (if a wrongly stayed: 600/3=200.0)
    """
    day1 = datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc)
    bar_a = _flat_bar(day1, 100.0, 1.0, symbol="BTC-USDT", source_feed="okx_demo")
    bar_b = _flat_bar(day1 + timedelta(hours=12), 200.0, 1.0, symbol="BTC-USDT", source_feed="okx_demo")
    bar_c = _flat_bar(day1 + timedelta(hours=25), 300.0, 1.0, symbol="BTC-USDT", source_feed="okx_demo")

    result = engine.vwap([bar_a, bar_b, bar_c])

    assert result == pytest.approx([100.0, 150.0, 250.0])


def test_vwap_empty_input_returns_empty_list() -> None:
    assert engine.vwap([]) == []


def test_vwap_none_when_window_has_zero_volume() -> None:
    base = datetime(2026, 9, 1, tzinfo=timezone.utc)
    bar = _flat_bar(base, 100.0, 0.0, source_feed="alpaca_iex")
    assert engine.vwap([bar]) == [None]


# --------------------------------------------------------------------------- #
# compute() -- the public entry point, ties everything together
# --------------------------------------------------------------------------- #


def test_compute_returns_an_indicator_frame_aligned_to_bars() -> None:
    base = datetime(2026, 9, 1, tzinfo=timezone.utc)
    bars = [_bar(base + timedelta(minutes=i), high=102 + i, low=100 + i, close=101 + i, volume=100 + i) for i in range(30)]

    frame = engine.compute(bars)

    n = len(bars)
    assert len(frame.vwap) == n
    assert len(frame.ema9) == n
    assert len(frame.ema21) == n
    assert len(frame.rsi14) == n
    assert len(frame.obv) == n
    assert len(frame.adx14) == n
    assert len(frame.atr14) == n
    # Spot-check a couple of fields land on values, not all-None/all-zero.
    assert frame.ema9[-1] is not None
    assert frame.rsi14[-1] is not None


def test_compute_empty_bars_returns_empty_frame() -> None:
    frame = engine.compute([])
    assert frame.vwap == []
    assert frame.ema9 == []
    assert frame.obv == []
