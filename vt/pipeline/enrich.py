"""Concrete `EnrichmentFn` for the M002 -> M005 -> M006 -> M007 pipeline.

The pipeline (`vt.pipeline.runner`) takes the R5/R6 rubric inputs -- the
structural facts (prior close, opening-range high/low,
break-and-hold-on-retest) and relative-strength-vs-benchmark that no
upstream module produces -- via an injected `EnrichmentFn`. S024 shipped
the composition with a test-supplied fake enricher; this module is the
real one. It is the drop-in that closes the gap between "runs against
fakes" and "runs against a real trading session" (Pipeline_Reference.md
Sec5 item 1).

`make_enricher(feed, ...)` returns an `EnrichmentFn` closed over a live
`FeedProtocol`. The returned function, for one universe candidate:

  * Reads the indicator-derived fields (VWAP / EMA9 / EMA21 / RSI14 / OBV
    slope / ADX14 / ATR-expansion / raw ATR14 for the stop) straight off
    the `IndicatorFrame` the pipeline already computed on the symbol's
    5-min bars -- no recomputation.
  * Derives the R5 structural facts from those same bars: the prior
    session's close from daily bars, the opening-range high/low from the
    first `opening_range_minutes` of the current session, and a
    deterministic broke-and-held-on-retest heuristic.
  * Computes R6 relative strength as the symbol's trailing-`rs_window`
    return minus the benchmark's (SPY for equities, BTC-USDT for crypto),
    fetched through the same feed.

Design choices worth stating:

  * **Factory, not a bare function.** The `EnrichmentFn` signature is
    `(UniverseCandidate, bars, IndicatorFrame, Quote) -> Enrichment` -- it
    is deliberately NOT handed the feed, because the pipeline treats
    enrichment as a pure mapping over data it already fetched. The
    benchmark leg needs its own feed access, so a factory closes over the
    feed and returns the pure-looking mapping the pipeline wants. Tests
    inject a fake feed exactly as the pipeline does.

  * **Live price comes from the quote, not the last bar.** R1 ("price
    above VWAP") and the entry/sizing math want the current price, so
    `price` is the quote's `last` (or bid/ask mid), not `bars[-1].close`.
    The VWAP it is compared against is bar-derived -- that
    slightly-different-oranges gap is exactly what a live momentum screen
    tolerates; it is answering "is price *now* above session VWAP." Entry
    at the *retest* level (Strategy_Spec.md Sec4) is a later refinement;
    the first dry-run prices entry at the current quote.

  * **Long-only.** Strategy_Spec.md Sec3 Direction: the paper phase is
    long-only. `side` is always "long". Shorts add four failure modes
    (borrow, locate, hard-to-borrow, squeeze) unrelated to whether the
    rubric works; revisit after the M-8 gate.

  * **Fail loud on missing data, not on an absent setup.** A candidate
    whose price is simply below its prior close, or that never broke its
    opening range, is a legitimate low score (R5=0/1) -- returned, not
    raised. But genuinely insufficient data (empty bars, indicator warmup
    never satisfied, no prior daily close, empty benchmark series) raises
    `EnrichmentError`. This matches the codebase's fail-loud convention
    (`vt.data.feed.DataFeedError` never swallows into an empty result): in
    a supervised dry-run you want to hear that your data pipeline is
    incomplete before you trade on it.

    Because `vt.pipeline.runner` calls `enrich()` without per-symbol
    isolation, a raise aborts the whole pass. That is acceptable for the
    supervised first runs this unblocks; isolating one bad symbol from the
    rest of the scan is a documented follow-up on the pipeline, not this
    module's job.

Full field contract mirrors the `Enrichment` dataclass in
`vt.pipeline.runner`. Tests: `vt/tests/test_enrich.py`.
"""

from __future__ import annotations

import math
from typing import Sequence

from vt.data.feed import Bar, Quote
from vt.indicators.engine import IndicatorFrame
from vt.pipeline.runner import Enrichment, EnrichmentFn, FeedProtocol
from vt.universe.screen import Candidate as UniverseCandidate

# Strategy_Spec.md Sec3 R5 ("Structure") / Sec5 ("No trading the first 5
# minutes") -- the opening range is the first 15 min of the session by
# default (classic ORB window). A 15-min window over 5-min bars is the
# first three session bars; a 5-min bar's high is already the max of its
# constituent 1-min highs, so this equals the 1-min opening range exactly.
DEFAULT_OPENING_RANGE_MINUTES = 15.0

# Strategy_Spec.md Sec3 R6 ("Relative strength ... 30-min").
DEFAULT_RS_WINDOW_MINUTES = 30.0

# Lookback (in bars) for OBV-slope sign and ATR-expansion. Short by
# design -- these are "is it rising *now*" reads, not long trends.
DEFAULT_SLOPE_LOOKBACK_BARS = 3

# A dip whose low comes within this band of the opening-range high counts
# as a retest touch. 0.1% is a first-cut; the retest read is the field
# most likely to want tick-level data later (see module docstring).
DEFAULT_RETEST_TOLERANCE_PCT = 0.001

_EQUITY_BENCHMARK = "SPY"
_CRYPTO_BENCHMARK = "BTC-USDT"
_CRYPTO_VENUE = "okx"
_LONG_ONLY_SIDE = "long"  # Strategy_Spec.md Sec3 Direction


class EnrichmentError(RuntimeError):
    """A candidate could not be enriched because required data was missing
    or insufficient -- as distinct from a candidate whose setup is simply
    absent (which scores low but enriches fine). Deliberately loud: a
    silent empty/zeroed enrichment would let a data-pipeline gap masquerade
    as a no-trade day.
    """


# --------------------------------------------------------------------------- #
# Small pure helpers (no I/O)
# --------------------------------------------------------------------------- #


def _last_present(seq: Sequence[float | None], *, name: str) -> float:
    """The most recent non-None value in an indicator series.

    Raises `EnrichmentError` when the series is entirely None (indicator
    warmup was never satisfied -- not enough bars).
    """
    for value in reversed(seq):
        if value is not None:
            return value
    raise EnrichmentError(
        f"indicator '{name}' has no computed value -- insufficient warmup bars"
    )


def _last_two_present(seq: Sequence[float | None]) -> tuple[float, float] | None:
    """The two most recent non-None values as `(older, newer)`, or None
    when fewer than two exist."""
    found: list[float] = []
    for value in reversed(seq):
        if value is not None:
            found.append(value)
            if len(found) == 2:
                return found[1], found[0]  # (older, newer)
    return None


def _is_rising(seq: Sequence[float | None]) -> bool:
    """True iff the two most recent computed values are strictly
    increasing. Unknown slope (fewer than two computed values) reads as
    not-rising -- conservative, so R1 can't reach its top score on data
    we can't actually confirm is trending up."""
    pair = _last_two_present(seq)
    return pair is not None and pair[1] > pair[0]


def _infer_interval_minutes(bars: Sequence[Bar], *, default: float = 5.0) -> float:
    """Bar interval in minutes, inferred from the smallest positive gap
    between consecutive bar timestamps.

    The smallest gap is the true bar spacing; session/overnight gaps are
    larger, never smaller, so they don't distort it. Falls back to
    `default` when there aren't two bars to measure.
    """
    if len(bars) < 2:
        return default
    deltas = [
        (bars[i].time - bars[i - 1].time).total_seconds()
        for i in range(1, len(bars))
    ]
    positive = [d for d in deltas if d > 0]
    if not positive:
        return default
    return min(positive) / 60.0


def _session_bars(bars: Sequence[Bar]) -> list[Bar]:
    """Bars sharing the latest bar's UTC calendar day.

    UTC-calendar-day session boundary matches M003's VWAP reset convention
    (`vt.indicators.engine`) -- this module does not know exchange session
    hours or holidays (that is M004's job), and deliberately stays
    consistent with the VWAP the rubric compares price against.
    """
    last_day = bars[-1].time.date()
    return [b for b in bars if b.time.date() == last_day]


def _opening_range(
    session: Sequence[Bar], *, minutes: float, interval_minutes: float
) -> tuple[float, float, int]:
    """(high, low, n_bars) of the first `minutes` of the session."""
    n = max(1, math.ceil(minutes / interval_minutes))
    window = session[:n]
    high = max(b.high for b in window)
    low = min(b.low for b in window)
    return high, low, len(window)


def _broke_and_held(
    session: Sequence[Bar], *, or_high: float, or_n: int, tolerance_pct: float
) -> tuple[bool, bool]:
    """(broke_opening_range_high, held_on_retest) over the post-opening
    bars.

    Broke: any bar after the opening range printed a high above `or_high`.
    Held: after that first breakout bar, a later bar dipped to within
    `tolerance_pct` of `or_high` (the retest) yet closed back at or above
    it (held). This is a deterministic bar-level proxy for Strategy_Spec.md
    Sec4's "limit order at the retest of the breakout level ... held"; the
    true read wants tick data, noted in the module docstring.
    """
    post = session[or_n:]
    seen_break = False
    broke = False
    held = False
    for bar in post:
        if not seen_break:
            if bar.high > or_high:
                seen_break = True
                broke = True
            continue
        touched_retest = bar.low <= or_high * (1.0 + tolerance_pct)
        closed_above = bar.close >= or_high
        if touched_retest and closed_above:
            held = True
            break
    return broke, held


def _quote_price(quote: Quote, *, fallback: float) -> float:
    """Current tradeable price from a quote: `last` if present, else the
    bid/ask mid, else a single side, else `fallback` (the last bar close).

    Falls back rather than raising -- an unusable quote when we still have
    a bar close is a stale-quote problem the gate's freshness step (M006
    step 7) is there to catch, not a reason this module can't produce a
    price at all.
    """
    if quote.last is not None and quote.last > 0:
        return float(quote.last)
    if quote.bid is not None and quote.ask is not None and quote.bid > 0 and quote.ask > 0:
        return (float(quote.bid) + float(quote.ask)) / 2.0
    if quote.bid is not None and quote.bid > 0:
        return float(quote.bid)
    if quote.ask is not None and quote.ask > 0:
        return float(quote.ask)
    return fallback


def _obv_slope(obv: Sequence[float], *, lookback: int) -> float:
    """OBV change over the last `lookback` bars -- sign is what R3 reads
    (positive rising, ~0 flat, negative falling)."""
    if len(obv) < 2:
        return 0.0
    k = min(lookback, len(obv) - 1)
    return obv[-1] - obv[-1 - k]


def _atr_expanding(atr14: Sequence[float | None], *, lookback: int) -> bool:
    """True iff the latest computed ATR exceeds the ATR `lookback` computed
    values earlier. Fewer than two computed values reads as not-expanding
    (conservative, same reasoning as `_is_rising`)."""
    present = [v for v in atr14 if v is not None]
    if len(present) < 2:
        return False
    earlier = present[-1 - min(lookback, len(present) - 1)]
    return present[-1] > earlier


def _pct_return(bars: Sequence[Bar], *, window_bars: int) -> float:
    """Percent close-to-close return over the trailing `window_bars`
    bars. Uses the earliest available close when the series is shorter than
    the window (a just-opened session), rather than raising."""
    if not bars:
        raise EnrichmentError("empty bar series for return window")
    if len(bars) >= window_bars + 1:
        start = bars[-(window_bars + 1)].close
    else:
        start = bars[0].close
    if start <= 0:
        raise EnrichmentError("non-positive reference close in return window")
    return (bars[-1].close / start - 1.0) * 100.0


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def make_enricher(
    feed: FeedProtocol,
    *,
    opening_range_minutes: float = DEFAULT_OPENING_RANGE_MINUTES,
    rs_window_minutes: float = DEFAULT_RS_WINDOW_MINUTES,
    slope_lookback_bars: int = DEFAULT_SLOPE_LOOKBACK_BARS,
    retest_tolerance_pct: float = DEFAULT_RETEST_TOLERANCE_PCT,
    equity_benchmark: str = _EQUITY_BENCHMARK,
    crypto_benchmark: str = _CRYPTO_BENCHMARK,
) -> EnrichmentFn:
    """Build an `EnrichmentFn` closed over `feed`.

    The returned function is the concrete replacement for the test-supplied
    fake enricher `vt.pipeline.runner` accepts -- see this module's
    docstring for exactly which fields it derives from where. `feed` is the
    same `FeedProtocol` the pipeline uses (`vt.data.feed` in production, a
    fake in tests); it is used here only to fetch prior-session daily bars
    and the benchmark series for relative strength.
    """

    def enrich(
        uc: UniverseCandidate,
        bars: Sequence[Bar],
        indicators: IndicatorFrame,
        quote: Quote,
    ) -> Enrichment:
        if not bars:
            raise EnrichmentError(f"no bars for {uc.symbol} -- cannot enrich")

        ref_time = bars[-1].time
        interval = _infer_interval_minutes(bars)

        vwap = _last_present(indicators.vwap, name="vwap")
        ema9 = _last_present(indicators.ema9, name="ema9")
        ema21 = _last_present(indicators.ema21, name="ema21")
        rsi14 = _last_present(indicators.rsi14, name="rsi14")
        adx14 = _last_present(indicators.adx14, name="adx14")
        atr14 = _last_present(indicators.atr14, name="atr14")

        price = _quote_price(quote, fallback=bars[-1].close)

        session = _session_bars(bars)
        or_high, or_low, or_n = _opening_range(
            session, minutes=opening_range_minutes, interval_minutes=interval
        )
        broke, held = _broke_and_held(
            session, or_high=or_high, or_n=or_n, tolerance_pct=retest_tolerance_pct
        )

        prior_close = _prior_close(feed, uc.symbol, ref_time=ref_time)

        benchmark = crypto_benchmark if uc.venue == _CRYPTO_VENUE else equity_benchmark
        rs_pct = _relative_strength_pct(
            feed,
            uc,
            bars,
            ref_time=ref_time,
            window_minutes=rs_window_minutes,
            interval_minutes=interval,
            benchmark_symbol=benchmark,
        )

        return Enrichment(
            price=price,
            vwap=vwap,
            ema9=ema9,
            ema21=ema21,
            ema9_rising=_is_rising(indicators.ema9),
            ema21_rising=_is_rising(indicators.ema21),
            rsi14=rsi14,
            rvol=uc.time_of_day_rvol,
            obv_slope=_obv_slope(indicators.obv, lookback=slope_lookback_bars),
            adx14=adx14,
            atr_expanding=_atr_expanding(indicators.atr14, lookback=slope_lookback_bars),
            prior_close=prior_close,
            opening_range_high=or_high,
            opening_range_low=or_low,
            broke_opening_range_high=broke,
            held_on_retest=held,
            relative_strength_pct=rs_pct,
            side=_LONG_ONLY_SIDE,
            atr_for_stop=atr14,
        )

    return enrich


# --------------------------------------------------------------------------- #
# Feed-touching helpers (kept out of the closure body for testability)
# --------------------------------------------------------------------------- #


def _prior_close(feed: FeedProtocol, symbol: str, *, ref_time) -> float:
    """The close of the most recent daily session strictly before
    `ref_time`'s date. Raises `EnrichmentError` when no such bar exists."""
    daily = feed.get_bars(symbol, "1d", end=ref_time, limit=10)
    prior = [b for b in daily if b.time.date() < ref_time.date()]
    if not prior:
        raise EnrichmentError(
            f"no prior daily bar for {symbol} before {ref_time.date()} "
            "-- cannot establish prior close"
        )
    prior.sort(key=lambda b: b.time)
    return prior[-1].close


def _relative_strength_pct(
    feed: FeedProtocol,
    uc: UniverseCandidate,
    symbol_bars: Sequence[Bar],
    *,
    ref_time,
    window_minutes: float,
    interval_minutes: float,
    benchmark_symbol: str,
) -> float:
    """Symbol trailing-window return minus benchmark trailing-window
    return, in percent. Positive = outperforming (R6 rewards > +1%).

    A candidate that *is* the benchmark (e.g. screening BTC-USDT itself)
    has zero relative strength by definition -- returned without a fetch.
    """
    window_bars = max(1, round(window_minutes / interval_minutes))
    symbol_ret = _pct_return(symbol_bars, window_bars=window_bars)

    if uc.symbol == benchmark_symbol:
        return 0.0

    bench_bars = feed.get_bars(
        benchmark_symbol, "5m", end=ref_time, limit=window_bars * 3 + 10
    )
    if not bench_bars:
        raise EnrichmentError(
            f"no benchmark bars for {benchmark_symbol} -- cannot compute relative strength"
        )
    bench_ret = _pct_return(bench_bars, window_bars=window_bars)
    return symbol_ret - bench_ret
