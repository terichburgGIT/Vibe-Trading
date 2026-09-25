"""Tests for the concrete `EnrichmentFn` (`vt.pipeline.enrich`).

Two layers, matching the module:
  * The pure helpers (`_opening_range`, `_broke_and_held`, `_pct_return`,
    `_quote_price`, slope/interval readers) -- hand-verified exact values,
    no I/O.
  * The `make_enricher` factory over a fake feed -- proves the closure
    pulls indicator fields off the passed-in `IndicatorFrame`, derives R5
    structure from the bars, fetches prior close + benchmark through the
    feed, and is a drop-in for `run_once`.

The indicator *math* (VWAP/EMA/RSI/OBV/ADX/ATR values) is M003's contract,
golden-tested in `test_engine.py`; here we only assert the enricher
surfaces the right (last-computed) element of each series, which is not
circular -- it tests the plumbing, not the arithmetic.

Run with: pytest vt/tests -m unit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pytest

from vt.data.feed import Bar, DataFeedError, Quote
from vt.exec.adapter import Position
from vt.gate.calendar import GateState
from vt.indicators import engine
from vt.pipeline import runner
from vt.pipeline.enrich import (
    EnrichmentError,
    _atr_expanding,
    _broke_and_held,
    _infer_interval_minutes,
    _is_rising,
    _last_present,
    _last_two_present,
    _obv_slope,
    _opening_range,
    _pct_return,
    _quote_price,
    _session_bars,
    make_enricher,
)
from vt.risk.gate import BreakerState
from vt.signal import rubric
from vt.universe.screen import Candidate as UniverseCandidate

pytestmark = pytest.mark.unit


DAY = datetime(2026, 9, 9, 13, 30, 0, tzinfo=timezone.utc)  # a session start


def _bar(
    dt: datetime,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 1000.0,
    symbol: str = "AAPL",
    source_feed: str = "alpaca_iex",
) -> Bar:
    return Bar(
        time=dt,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        symbol=symbol,
        source_feed=source_feed,
    )


def _five_min_series(closes: list[float], *, start: datetime = DAY, symbol: str = "AAPL",
                     source_feed: str = "alpaca_iex") -> list[Bar]:
    """Bars 5 minutes apart. Each bar's high/low straddle its close by 0.1
    unless overridden; open == prior close (or first close)."""
    bars: list[Bar] = []
    prev = closes[0]
    for i, c in enumerate(closes):
        t = start + timedelta(minutes=5 * i)
        bars.append(_bar(t, open_=prev, high=max(prev, c) + 0.1, low=min(prev, c) - 0.1,
                         close=c, symbol=symbol, source_feed=source_feed))
        prev = c
    return bars


# --------------------------------------------------------------------------- #
# _last_present / _last_two_present / _is_rising
# --------------------------------------------------------------------------- #


def test_last_present_returns_final_non_none() -> None:
    assert _last_present([None, 1.0, 2.0, None, 3.0], name="x") == 3.0


def test_last_present_raises_when_all_none() -> None:
    with pytest.raises(EnrichmentError, match="insufficient warmup"):
        _last_present([None, None], name="rsi14")


def test_last_two_present_gives_older_then_newer() -> None:
    assert _last_two_present([1.0, None, 2.0, 3.0]) == (2.0, 3.0)


def test_last_two_present_none_when_fewer_than_two() -> None:
    assert _last_two_present([None, 5.0]) is None


def test_is_rising_true_when_newer_greater() -> None:
    assert _is_rising([1.0, 2.0, 3.0]) is True


def test_is_rising_false_when_flat_or_falling() -> None:
    assert _is_rising([3.0, 3.0]) is False
    assert _is_rising([3.0, 2.0]) is False


def test_is_rising_false_when_slope_unknown() -> None:
    # Only one computed value -> can't confirm an uptrend -> conservative False.
    assert _is_rising([None, 5.0]) is False


# --------------------------------------------------------------------------- #
# _infer_interval_minutes
# --------------------------------------------------------------------------- #


def test_infer_interval_from_five_min_bars() -> None:
    bars = _five_min_series([100.0, 101.0, 102.0])
    assert _infer_interval_minutes(bars) == pytest.approx(5.0)


def test_infer_interval_ignores_overnight_gap() -> None:
    # Two same-day 5-min bars then a bar the next day; the smallest gap
    # (5 min) is the interval, not the overnight jump.
    b1 = _bar(DAY, open_=100, high=100.1, low=99.9, close=100.0)
    b2 = _bar(DAY + timedelta(minutes=5), open_=100, high=100.1, low=99.9, close=100.0)
    b3 = _bar(DAY + timedelta(days=1), open_=100, high=100.1, low=99.9, close=100.0)
    assert _infer_interval_minutes([b1, b2, b3]) == pytest.approx(5.0)


def test_infer_interval_falls_back_with_one_bar() -> None:
    assert _infer_interval_minutes([_bar(DAY, open_=1, high=1, low=1, close=1)]) == 5.0


# --------------------------------------------------------------------------- #
# _opening_range
# --------------------------------------------------------------------------- #


def test_opening_range_is_first_15_min_over_5_min_bars() -> None:
    # First 3 bars (15 min) define the range; later bars don't widen it.
    bars = [
        _bar(DAY + timedelta(minutes=0), open_=100, high=101, low=99, close=100),
        _bar(DAY + timedelta(minutes=5), open_=100, high=102, low=99.5, close=101),
        _bar(DAY + timedelta(minutes=10), open_=101, high=101.5, low=100, close=101),
        _bar(DAY + timedelta(minutes=15), open_=101, high=110, low=95, close=108),  # excluded
    ]
    high, low, n = _opening_range(bars, minutes=15.0, interval_minutes=5.0)
    assert n == 3
    assert high == 102.0  # max high of first three bars
    assert low == 99.0    # min low of first three bars


def test_opening_range_window_always_at_least_one_bar() -> None:
    bars = [_bar(DAY, open_=100, high=101, low=99, close=100)]
    high, low, n = _opening_range(bars, minutes=15.0, interval_minutes=5.0)
    assert (high, low, n) == (101.0, 99.0, 1)


# --------------------------------------------------------------------------- #
# _broke_and_held
# --------------------------------------------------------------------------- #


def _session_with_or(or_bars: list[Bar], post_bars: list[Bar]) -> list[Bar]:
    return or_bars + post_bars


def test_broke_and_held_true_on_breakout_then_holding_retest() -> None:
    or_bars = [_bar(DAY, open_=100, high=105, low=99, close=104)]  # OR high 105
    post = [
        _bar(DAY + timedelta(minutes=5), open_=104, high=107, low=105.5, close=106),  # breakout
        _bar(DAY + timedelta(minutes=10), open_=106, high=106.5, low=104.9, close=105.3),  # retest holds
    ]
    broke, held = _broke_and_held(_session_with_or(or_bars, post), or_high=105.0, or_n=1,
                                  tolerance_pct=0.001)
    assert broke is True
    assert held is True


def test_broke_but_not_held_when_retest_closes_below_or_high() -> None:
    or_bars = [_bar(DAY, open_=100, high=105, low=99, close=104)]
    post = [
        _bar(DAY + timedelta(minutes=5), open_=104, high=107, low=105.5, close=106),  # breakout
        _bar(DAY + timedelta(minutes=10), open_=106, high=106, low=103, close=104.0),  # fails retest
    ]
    broke, held = _broke_and_held(_session_with_or(or_bars, post), or_high=105.0, or_n=1,
                                  tolerance_pct=0.001)
    assert broke is True
    assert held is False


def test_no_break_means_neither_broke_nor_held() -> None:
    or_bars = [_bar(DAY, open_=100, high=105, low=99, close=104)]
    post = [
        _bar(DAY + timedelta(minutes=5), open_=104, high=104.9, low=103, close=104.2),
        _bar(DAY + timedelta(minutes=10), open_=104, high=104.5, low=103, close=104.0),
    ]
    broke, held = _broke_and_held(_session_with_or(or_bars, post), or_high=105.0, or_n=1,
                                  tolerance_pct=0.001)
    assert broke is False
    assert held is False


def test_breakout_bar_is_not_its_own_retest() -> None:
    # A single post bar that both breaks and dips can't count as its own
    # retest -- the retest must be a later, distinct bar.
    or_bars = [_bar(DAY, open_=100, high=105, low=99, close=104)]
    post = [_bar(DAY + timedelta(minutes=5), open_=104, high=107, low=104.9, close=105.2)]
    broke, held = _broke_and_held(_session_with_or(or_bars, post), or_high=105.0, or_n=1,
                                  tolerance_pct=0.001)
    assert broke is True
    assert held is False


# --------------------------------------------------------------------------- #
# _quote_price
# --------------------------------------------------------------------------- #


def _quote(symbol="AAPL", *, bid=None, ask=None, last=None, t=DAY, source_feed="fake") -> Quote:
    return Quote(symbol=symbol, bid=bid, ask=ask, last=last, time=t, source_feed=source_feed)


def test_quote_price_prefers_last() -> None:
    assert _quote_price(_quote(bid=99, ask=101, last=100.5), fallback=42.0) == 100.5


def test_quote_price_uses_mid_when_no_last() -> None:
    assert _quote_price(_quote(bid=99, ask=101, last=None), fallback=42.0) == 100.0


def test_quote_price_falls_back_to_last_bar_close_when_quote_unusable() -> None:
    assert _quote_price(_quote(bid=None, ask=None, last=None), fallback=42.0) == 42.0


# --------------------------------------------------------------------------- #
# _obv_slope / _atr_expanding
# --------------------------------------------------------------------------- #


def test_obv_slope_sign_tracks_direction() -> None:
    assert _obv_slope([0.0, 10.0, 20.0, 30.0], lookback=3) == 30.0
    assert _obv_slope([30.0, 20.0, 10.0, 0.0], lookback=3) == -30.0
    assert _obv_slope([5.0], lookback=3) == 0.0


def test_atr_expanding_compares_latest_to_earlier() -> None:
    assert _atr_expanding([None, 1.0, 1.5, 2.0], lookback=3) is True
    assert _atr_expanding([None, 2.0, 1.5, 1.0], lookback=3) is False
    assert _atr_expanding([None, 1.0], lookback=3) is False  # <2 computed


# --------------------------------------------------------------------------- #
# _pct_return
# --------------------------------------------------------------------------- #


def test_pct_return_over_full_window() -> None:
    # 7 bars, window 6 -> close[-1]/close[-7] - 1 = 110/100 - 1 = 10%.
    bars = _five_min_series([100, 101, 102, 103, 104, 105, 110])
    assert _pct_return(bars, window_bars=6) == pytest.approx(10.0)


def test_pct_return_uses_earliest_when_window_longer_than_series() -> None:
    bars = _five_min_series([100, 105])  # only 2 bars, window 6
    assert _pct_return(bars, window_bars=6) == pytest.approx(5.0)


def test_pct_return_raises_on_empty() -> None:
    with pytest.raises(EnrichmentError):
        _pct_return([], window_bars=6)


# --------------------------------------------------------------------------- #
# Fake feed + factory
# --------------------------------------------------------------------------- #


@dataclass
class FakeFeed:
    bars_by_key: dict[tuple[str, str], list[Bar]] = field(default_factory=dict)
    quote_by_symbol: dict[str, Quote] = field(default_factory=dict)

    def get_bars(self, symbol, timeframe="1d", *, start=None, end=None, limit=90):  # noqa: ARG002
        return list(self.bars_by_key.get((symbol, timeframe), []))

    def get_quote(self, symbol):
        return self.quote_by_symbol[symbol]


def _uc(symbol="AAPL", venue="alpaca", rvol=3.5) -> UniverseCandidate:
    return UniverseCandidate(
        symbol=symbol, venue=venue, time_of_day_rvol=rvol,
        dollar_volume_today=30_000_000.0, price=100.0, avg_spread_pct=0.05, atr14_pct=2.0,
    )


def _rising_session(symbol="AAPL", source_feed="alpaca_iex") -> list[Bar]:
    """40 gently rising 5-min bars so every indicator warms up and the
    trend reads up."""
    closes = [100.0 + i * 0.2 for i in range(40)]
    return _five_min_series(closes, symbol=symbol, source_feed=source_feed)


def test_enricher_surfaces_last_computed_indicator_values() -> None:
    bars = _rising_session()
    frame = engine.compute(bars)
    feed = FakeFeed(
        bars_by_key={
            ("AAPL", "1d"): [_bar(DAY - timedelta(days=1), open_=98, high=99, low=97, close=98.0)],
            ("SPY", "5m"): _five_min_series([400.0] * 10, symbol="SPY"),
        },
        quote_by_symbol={"AAPL": _quote(bid=107.9, ask=108.1, last=108.0)},
    )
    enrich = make_enricher(feed)

    e = enrich(_uc(), bars, frame, feed.get_quote("AAPL"))

    assert e.vwap == _last_present(frame.vwap, name="vwap")
    assert e.ema9 == _last_present(frame.ema9, name="ema9")
    assert e.ema21 == _last_present(frame.ema21, name="ema21")
    assert e.rsi14 == _last_present(frame.rsi14, name="rsi14")
    assert e.adx14 == _last_present(frame.adx14, name="adx14")
    assert e.atr_for_stop == _last_present(frame.atr14, name="atr14")
    # rvol comes from the universe candidate, not recomputed.
    assert e.rvol == 3.5
    # Long-only paper phase.
    assert e.side == "long"
    # Live price is the quote's, not the last bar close.
    assert e.price == 108.0


def test_enricher_reads_prior_close_from_daily_bars() -> None:
    bars = _rising_session()
    frame = engine.compute(bars)
    feed = FakeFeed(
        bars_by_key={
            ("AAPL", "1d"): [
                _bar(DAY - timedelta(days=2), open_=90, high=91, low=89, close=90.0),
                _bar(DAY - timedelta(days=1), open_=95, high=96, low=94, close=95.5),  # most recent prior
            ],
            ("SPY", "5m"): _five_min_series([400.0] * 10, symbol="SPY"),
        },
        quote_by_symbol={"AAPL": _quote(bid=107.9, ask=108.1, last=108.0)},
    )
    e = make_enricher(feed)(_uc(), bars, frame, feed.get_quote("AAPL"))
    assert e.prior_close == 95.5


def test_enricher_raises_when_no_prior_daily_bar() -> None:
    bars = _rising_session()
    frame = engine.compute(bars)
    feed = FakeFeed(
        bars_by_key={("AAPL", "1d"): [], ("SPY", "5m"): _five_min_series([400.0] * 10, symbol="SPY")},
        quote_by_symbol={"AAPL": _quote(bid=107.9, ask=108.1, last=108.0)},
    )
    with pytest.raises(EnrichmentError, match="prior daily bar"):
        make_enricher(feed)(_uc(), bars, frame, feed.get_quote("AAPL"))


def test_enricher_raises_on_empty_bars() -> None:
    feed = FakeFeed()
    frame = engine.compute([])
    with pytest.raises(EnrichmentError, match="no bars"):
        make_enricher(feed)(_uc(), [], frame, _quote())


def test_relative_strength_is_symbol_minus_benchmark() -> None:
    # Symbol +10% over the trailing 30 min, benchmark flat -> RS = +10%.
    sym_bars = _five_min_series([100, 101, 102, 103, 104, 105, 110])  # 7 bars, +10% over 6
    frame = engine.compute(_rising_session())  # frame only needs to warm up; RS uses sym_bars
    feed = FakeFeed(
        bars_by_key={
            ("AAPL", "1d"): [_bar(DAY - timedelta(days=1), open_=98, high=99, low=97, close=98.0)],
            ("SPY", "5m"): _five_min_series([400.0] * 8, symbol="SPY"),  # flat benchmark
        },
        quote_by_symbol={"AAPL": _quote(bid=109.9, ask=110.1, last=110.0)},
    )
    e = make_enricher(feed)(_uc(), sym_bars, frame, feed.get_quote("AAPL"))
    assert e.relative_strength_pct == pytest.approx(10.0)


def test_relative_strength_zero_against_self() -> None:
    # Screening the benchmark itself: RS vs itself is 0, no fetch needed.
    btc_bars = _five_min_series([100, 101, 102, 103, 104, 105, 110], symbol="BTC-USDT",
                                source_feed="okx_demo")
    frame = engine.compute(_rising_session(symbol="BTC-USDT", source_feed="okx_demo"))
    feed = FakeFeed(
        bars_by_key={("BTC-USDT", "1d"): [_bar(DAY - timedelta(days=1), open_=98, high=99,
                                               low=97, close=98.0, symbol="BTC-USDT",
                                               source_feed="okx_demo")]},
        quote_by_symbol={"BTC-USDT": _quote(symbol="BTC-USDT", last=110.0)},
    )
    e = make_enricher(feed)(_uc(symbol="BTC-USDT", venue="okx"), btc_bars, frame,
                            feed.get_quote("BTC-USDT"))
    assert e.relative_strength_pct == 0.0


def test_benchmark_selected_by_venue() -> None:
    # Crypto candidate must pull BTC-USDT as the benchmark, not SPY.
    sym_bars = _five_min_series([100, 100, 100, 100, 100, 100, 101], symbol="ETH-USDT",
                                source_feed="okx_demo")
    frame = engine.compute(_rising_session(symbol="ETH-USDT", source_feed="okx_demo"))
    feed = FakeFeed(
        bars_by_key={
            ("ETH-USDT", "1d"): [_bar(DAY - timedelta(days=1), open_=98, high=99, low=97,
                                      close=98.0, symbol="ETH-USDT", source_feed="okx_demo")],
            ("BTC-USDT", "5m"): _five_min_series([200.0] * 8, symbol="BTC-USDT",
                                                 source_feed="okx_demo"),  # flat
        },
        quote_by_symbol={"ETH-USDT": _quote(symbol="ETH-USDT", last=101.0)},
    )
    e = make_enricher(feed)(_uc(symbol="ETH-USDT", venue="okx"), sym_bars, frame,
                            feed.get_quote("ETH-USDT"))
    # ETH +1% (100->101 over 6 bars), BTC flat -> RS ~ +1%.
    assert e.relative_strength_pct == pytest.approx(1.0)


def test_enricher_raises_when_benchmark_series_empty() -> None:
    bars = _rising_session()
    frame = engine.compute(bars)
    feed = FakeFeed(
        bars_by_key={
            ("AAPL", "1d"): [_bar(DAY - timedelta(days=1), open_=98, high=99, low=97, close=98.0)],
            ("SPY", "5m"): [],  # benchmark missing
        },
        quote_by_symbol={"AAPL": _quote(last=108.0)},
    )
    with pytest.raises(EnrichmentError, match="benchmark"):
        make_enricher(feed)(_uc(), bars, frame, feed.get_quote("AAPL"))


# --------------------------------------------------------------------------- #
# R5 structure end to end through the enricher
# --------------------------------------------------------------------------- #


def _break_and_hold_5m_bars() -> tuple[list[Bar], float]:
    """5-min bars whose own OHLC shows a break-and-hold directly (i.e. the
    5-min bar-level proxy can detect it without any 1-min data) -- shared
    by the "no 1m data available" test and the "1m feed errors" test,
    which should both land on the same proxy answer."""
    # First 3 bars: tight range 99.9-100.1 (OR high ~100.2 given the +0.1
    # straddle). Then a breakout and a holding retest.
    or_closes = [100.0, 100.0, 100.0]
    or_bars = _five_min_series(or_closes)  # highs ~100.1, lows ~99.9
    or_high = max(b.high for b in or_bars[:3])
    post = [
        _bar(DAY + timedelta(minutes=15), open_=100, high=or_high + 2, low=or_high + 0.5,
             close=or_high + 1.5),  # breakout
        _bar(DAY + timedelta(minutes=20), open_=or_high + 1.5, high=or_high + 1.6,
             low=or_high - 0.01, close=or_high + 0.5),  # retest, holds
    ]
    bars = or_bars + post
    # Pad with more rising bars so indicators warm up (append after the setup).
    tail = _five_min_series([or_high + 0.5 + i * 0.2 for i in range(40)],
                            start=DAY + timedelta(minutes=25))
    return bars + tail, or_high


def test_enricher_reports_break_and_hold_structure() -> None:
    """A session that opens in a tight range, breaks the OR high, and holds
    a retest must surface broke=True, held=True with an OR high/low taken
    from the first 15 min. No 1-minute bars are registered on the feed, so
    this also exercises the fallback-to-5m-proxy path in
    `_retest_confirmation` (the common case until every venue's 1m data is
    confirmed reachable)."""
    bars, or_high = _break_and_hold_5m_bars()
    frame = engine.compute(bars)

    feed = FakeFeed(
        bars_by_key={
            ("AAPL", "1d"): [_bar(DAY - timedelta(days=1), open_=98, high=99, low=97, close=98.0)],
            ("SPY", "5m"): _five_min_series([400.0] * 10, symbol="SPY"),
        },
        quote_by_symbol={"AAPL": _quote(last=bars[-1].close)},
    )
    e = make_enricher(feed)(_uc(), bars, frame, feed.get_quote("AAPL"))

    assert e.opening_range_high == pytest.approx(or_high)
    assert e.broke_opening_range_high is True
    assert e.held_on_retest is True
    # And the rubric reads that as R5=2 (price is well above prior close 98).
    score = rubric.score(_rubric_candidate_from(e))
    assert score.breakdown["R5"] == 2


class _RaisingMinuteFeed(FakeFeed):
    """A feed that can serve everything except 1-minute bars (simulates a
    venue/outage that doesn't support 1m data)."""

    def get_bars(self, symbol, timeframe="1d", *, start=None, end=None, limit=90):
        if timeframe == "1m":
            raise DataFeedError("simulated 1m outage")
        return super().get_bars(symbol, timeframe, start=start, end=end, limit=limit)


def test_enricher_falls_back_to_5m_proxy_when_1m_feed_errors() -> None:
    """Retest confirmation is a precision upgrade to one existing field,
    not a hard data requirement (see `_retest_confirmation`'s docstring)
    -- a `DataFeedError` fetching 1-minute bars must degrade to the 5-min
    proxy, not abort the whole enrichment."""
    bars, or_high = _break_and_hold_5m_bars()
    frame = engine.compute(bars)

    feed = _RaisingMinuteFeed(
        bars_by_key={
            ("AAPL", "1d"): [_bar(DAY - timedelta(days=1), open_=98, high=99, low=97, close=98.0)],
            ("SPY", "5m"): _five_min_series([400.0] * 10, symbol="SPY"),
        },
        quote_by_symbol={"AAPL": _quote(last=bars[-1].close)},
    )
    e = make_enricher(feed)(_uc(), bars, frame, feed.get_quote("AAPL"))

    assert e.opening_range_high == pytest.approx(or_high)
    assert e.broke_opening_range_high is True
    assert e.held_on_retest is True  # from the 5-min proxy fallback


def test_enricher_uses_1m_bars_to_catch_a_retest_the_5m_proxy_misses() -> None:
    """The exact gap AD016/S031 closes: a retest-and-hold that happens
    entirely inside the breakout's own 5-min candle is invisible to the
    5-min proxy (`_broke_and_held` skips the retest check on the same bar
    where the breakout is first seen -- see that function's docstring),
    but the real 1-minute bars show a genuine momentary hold at minute 16
    before price fades. This proves 1-minute data changes the answer, not
    just that it runs without error.
    """
    # Opening range: 15 flat one-minute bars, or_high ~= 100.1.
    minute_bars = [
        _bar(DAY + timedelta(minutes=i), open_=100.0, high=100.1, low=99.9, close=100.0)
        for i in range(15)
    ]
    # Breakout (m15), a momentary retest-and-hold (m16), then a fade that
    # never reclaims the level (m17-19) -- all inside one 5-min bucket.
    minute_bars += [
        _bar(DAY + timedelta(minutes=15), open_=100.0, high=102.0, low=101.9, close=101.95),
        _bar(DAY + timedelta(minutes=16), open_=101.95, high=101.95, low=100.05, close=100.12),
        _bar(DAY + timedelta(minutes=17), open_=100.12, high=100.12, low=99.5, close=99.6),
        _bar(DAY + timedelta(minutes=18), open_=99.6, high=99.65, low=99.4, close=99.5),
        _bar(DAY + timedelta(minutes=19), open_=99.5, high=99.55, low=99.3, close=99.4),
    ]
    # Next bucket: continues fading, never comes back near the level.
    minute_bars += [
        _bar(DAY + timedelta(minutes=20), open_=99.4, high=99.5, low=99.2, close=99.3),
        _bar(DAY + timedelta(minutes=21), open_=99.3, high=99.4, low=99.1, close=99.2),
        _bar(DAY + timedelta(minutes=22), open_=99.2, high=99.3, low=99.0, close=99.1),
        _bar(DAY + timedelta(minutes=23), open_=99.1, high=99.2, low=98.9, close=99.0),
        _bar(DAY + timedelta(minutes=24), open_=99.0, high=99.1, low=98.8, close=98.9),
    ]

    # The 5-min bars are a real OHLC roll-up of those same minutes (open
    # = first, high = max, low = min, close = last) -- not a fabricated
    # mismatch -- plus a declining tail (never re-crosses or_high, so it
    # can't accidentally trip the proxy) long enough for indicators to warm up.
    or_5m = _five_min_series([100.0, 100.0, 100.0])
    bucket_4 = _bar(DAY + timedelta(minutes=15), open_=100.0, high=102.0, low=99.3, close=99.4)
    bucket_5 = _bar(DAY + timedelta(minutes=20), open_=99.4, high=99.5, low=98.8, close=98.9)
    tail = _five_min_series([98.9 - i * 0.2 for i in range(40)],
                            start=DAY + timedelta(minutes=25))
    bars = or_5m + [bucket_4, bucket_5] + tail
    frame = engine.compute(bars)

    feed = FakeFeed(
        bars_by_key={
            ("AAPL", "1d"): [_bar(DAY - timedelta(days=1), open_=98, high=99, low=97, close=98.0)],
            ("SPY", "5m"): _five_min_series([400.0] * 10, symbol="SPY"),
            ("AAPL", "1m"): minute_bars,
        },
        quote_by_symbol={"AAPL": _quote(last=bars[-1].close)},
    )

    # Sanity check: the 5-min-only read genuinely misses this hold, so the
    # test is demonstrating a real gap, not a coincidence.
    session_5m = _session_bars(bars)
    or_high_5m, _, or_n_5m = _opening_range(session_5m, minutes=15.0, interval_minutes=5.0)
    _, held_5m_only = _broke_and_held(
        session_5m, or_high=or_high_5m, or_n=or_n_5m, tolerance_pct=0.001
    )
    assert held_5m_only is False

    e = make_enricher(feed)(_uc(), bars, frame, feed.get_quote("AAPL"))

    assert e.broke_opening_range_high is True
    assert e.held_on_retest is True  # only visible at 1-minute resolution


def _rubric_candidate_from(e: runner.Enrichment, symbol="AAPL", venue="alpaca") -> rubric.Candidate:
    return rubric.Candidate(
        symbol=symbol, venue=venue, price=e.price, vwap=e.vwap, ema9=e.ema9, ema21=e.ema21,
        ema9_rising=e.ema9_rising, ema21_rising=e.ema21_rising, rsi14=e.rsi14, rvol=e.rvol,
        obv_slope=e.obv_slope, adx14=e.adx14, atr_expanding=e.atr_expanding,
        prior_close=e.prior_close, opening_range_high=e.opening_range_high,
        opening_range_low=e.opening_range_low,
        broke_opening_range_high=e.broke_opening_range_high, held_on_retest=e.held_on_retest,
        relative_strength_pct=e.relative_strength_pct,
    )


# --------------------------------------------------------------------------- #
# Drop-in through the real pipeline (run_once)
# --------------------------------------------------------------------------- #


@dataclass
class _FakeBroker:
    venue: str
    positions_rows: list[Position] = field(default_factory=list)
    entry_calls: list[dict] = field(default_factory=list)
    stop_calls: list[dict] = field(default_factory=list)
    _seq: int = 0

    def submit_entry(self, *, symbol, side, size, limit_price, client_order_id):
        self._seq += 1
        self.entry_calls.append({"symbol": symbol, "client_order_id": client_order_id})
        signed = size if side == "long" else -size
        self.positions_rows.append(Position(venue=self.venue, symbol=symbol, quantity=signed))
        return f"entry-{self._seq}"

    def submit_stop(self, *, symbol, side, size, stop_price, client_order_id):
        self.stop_calls.append({"symbol": symbol})
        return f"stop-{self._seq}"

    def cancel(self, order_id):  # noqa: ARG002
        return None

    def close_position(self, symbol):
        self.positions_rows = [r for r in self.positions_rows if r.symbol != symbol]

    def positions(self):
        return list(self.positions_rows)


@dataclass
class _PipelineFeed:
    """Feed the whole pipeline drives: build_universe + get_bars(5m) for the
    symbol + daily + benchmark, plus quotes. Wraps a FakeFeed for the
    bar/quote routing and adds the `build_universe` hook run_once looks
    for."""

    inner: FakeFeed
    universe_by_venue: dict[str, list[UniverseCandidate]] = field(default_factory=dict)

    def build_universe(self, symbols, venue, *, asof):  # noqa: ARG002
        return list(self.universe_by_venue.get(venue, []))

    def get_bars(self, symbol, timeframe="1d", *, start=None, end=None, limit=90):
        return self.inner.get_bars(symbol, timeframe, start=start, end=end, limit=limit)

    def get_quote(self, symbol):
        return self.inner.get_quote(symbol)


def test_real_enricher_is_a_drop_in_for_run_once() -> None:
    """The whole point: swap the fake enricher for `make_enricher(feed)` and
    the composition still runs top to bottom, scoring the candidate off
    real bars with no exception."""
    bars = _rising_session()
    inner = FakeFeed(
        bars_by_key={
            ("AAPL", "5m"): bars,
            ("AAPL", "1d"): [_bar(DAY - timedelta(days=1), open_=98, high=99, low=97, close=98.0)],
            ("SPY", "5m"): _five_min_series([400.0] * 10, symbol="SPY"),
        },
        quote_by_symbol={"AAPL": _quote(bid=107.9, ask=108.1, last=108.0, t=DAY + timedelta(minutes=5 * 39))},
    )
    feed = _PipelineFeed(inner=inner, universe_by_venue={"alpaca": [_uc()]})
    broker = _FakeBroker(venue="alpaca")

    asof = bars[-1].time
    request = runner.PipelineRequest(
        symbols_by_venue={"alpaca": ["AAPL"]},
        asof=asof,
        now=asof,
        equity=100_000.0,
        breaker_state=BreakerState(),
        adapters={"alpaca": broker},
        calendar_state=GateState(state="NORMAL", multiplier=1.0, reasons=()),
        enrich=make_enricher(feed),
        feed=feed,
        dry_run=True,
    )

    outcome = runner.run_once(request)

    assert outcome.halted is False
    assert len(outcome.scores) == 1
    assert outcome.scores[0].symbol == "AAPL"
    # A gate decision was produced (dry-run: no broker/journal side effects).
    (row,) = outcome.decisions
    assert broker.entry_calls == []
    # The enrichment fed a real, in-range price/prior-close/benchmark; the
    # score is a real 0-12 with a full R1..R6 breakdown.
    assert set(outcome.scores[0].breakdown) == {"R1", "R2", "R3", "R4", "R5", "R6"}
