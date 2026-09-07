"""M002 — Universe Builder.

Daily screen producing <=20 eligible names per venue per
`Strategy_Spec.md` § 1. Net-new (no upstream equivalent).

Time-of-day-aware RVOL is the core requirement — the baseline is
volume-by-this-time-of-day over the trailing 20 sessions, never a
full-day average (that's the classic bug that makes the screen useless
before noon; T004 exists specifically to catch it).

Two layers, deliberately separated:
  - `screen()` is a pure function over `CandidateStats` — no I/O, no
    network, fully unit-testable (this is what T003/T004 exercise).
  - `build_universe()` is the thin orchestrator on top: given a caller-
    supplied watchlist, it pulls bars via `vt.data.feed` and calls
    `screen()`.

**Not yet enforced** (Strategy_Spec.md §1 filters with no data source
wired up yet — not silently assumed passing, just not implemented):
  - Equity float >= 10M shares
  - Earnings-blackout (+/-1 session)
  - Halt-in-last-5-sessions exclusion
  - Crypto listing-age >= 30 days exclusion
Leveraged-token and USDT-quote-only crypto exclusions *are* implemented
(name-based, no external data source needed).

Full contract: `03_Modules.md` § M002. Tests: `vt/tests/test_screen.py`
(T003, T004 in `06_Tests.md`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

from vt.data.feed import Bar

MAX_UNIVERSE_SIZE = 20  # Strategy_Spec.md §1: "Produces <=20 names per venue"

_EQUITY_VENUE = "alpaca"
_CRYPTO_VENUE = "okx"
_KNOWN_VENUES = (_EQUITY_VENUE, _CRYPTO_VENUE)

# Strategy_Spec.md §1a
_EQUITY_MIN_RVOL = 2.0
_EQUITY_MIN_DOLLAR_VOLUME = 20_000_000.0
_EQUITY_MIN_PRICE = 5.0
_EQUITY_MAX_PRICE = 500.0
_EQUITY_MAX_SPREAD_PCT = 0.15
_EQUITY_MIN_ATR_PCT = 1.5

# Strategy_Spec.md §1b
_CRYPTO_MIN_RVOL = 2.0
_CRYPTO_MIN_DOLLAR_VOLUME = 50_000_000.0
_CRYPTO_MIN_ATR_PCT = 1.5
_CRYPTO_LEVERAGED_TAGS = ("UP", "DOWN", "BULL", "BEAR")


@dataclass(frozen=True)
class CandidateStats:
    """Raw per-symbol stats for one screening pass — the input to `screen()`.

    Callers assemble this from `vt.data.feed` bars (today's + N-session
    history) plus a spread estimate. `avg_spread_pct` here is a single
    latest-quote spread, not a true intraday average — a known
    simplification until tick-level spread tracking exists.
    """

    symbol: str
    venue: str  # "alpaca" or "okx"
    time_of_day_rvol: float
    dollar_volume_today: float
    price: float
    avg_spread_pct: float
    atr14_pct: float


@dataclass(frozen=True)
class Candidate:
    """One symbol that passed every *implemented* Strategy_Spec.md §1 filter."""

    symbol: str
    venue: str
    time_of_day_rvol: float
    dollar_volume_today: float
    price: float
    avg_spread_pct: float
    atr14_pct: float


def volume_by_cutoff(bars: Sequence[Bar], session_date: date, cutoff: datetime) -> float:
    """Sum volume for bars on `session_date` at or before `cutoff`'s time-of-day.

    Both `bars` and `cutoff` are expected in UTC (per M001's AD003 UTC-
    internal rule) — this does not do exchange-timezone/DST conversion.
    Converting a "10:00 ET" session cutoff to the correct UTC time-of-day
    across DST is the caller's responsibility for now; that conversion
    isn't implemented here yet.
    """
    cutoff_time = cutoff.time()
    return sum(bar.volume for bar in bars if bar.time.date() == session_date and bar.time.time() <= cutoff_time)


def time_of_day_rvol(
    today_bars: Sequence[Bar],
    history_bars: Sequence[Bar],
    *,
    asof: datetime,
    lookback_sessions: int = 20,
) -> float:
    """RVOL against the N-session average of volume-by-this-time-of-day.

    This is the fix for the classic RVOL bug (Strategy_Spec.md §1a):
    comparing intraday cumulative volume to a *full-day* average makes
    everything look quiet before the close. The baseline here is volume
    traded by the same clock time on each of the last `lookback_sessions`
    prior sessions found in `history_bars`, never full-day volume.

    Raises:
        ValueError: No prior sessions found in `history_bars`, or the
            computed baseline is zero/negative (a ratio against that is
            meaningless, not just numerically awkward).
    """
    today_date = asof.date()
    today_volume = volume_by_cutoff(today_bars, today_date, asof)

    session_dates = sorted({b.time.date() for b in history_bars if b.time.date() != today_date})
    session_dates = session_dates[-lookback_sessions:]
    if not session_dates:
        raise ValueError("no history sessions available to build an RVOL baseline")

    baseline_volumes = [volume_by_cutoff(history_bars, d, asof) for d in session_dates]
    baseline = sum(baseline_volumes) / len(baseline_volumes)
    if baseline <= 0:
        raise ValueError("zero/negative RVOL baseline — cannot compute a meaningful ratio")
    return today_volume / baseline


def _passes_equity_filters(stats: CandidateStats) -> bool:
    return (
        stats.time_of_day_rvol >= _EQUITY_MIN_RVOL
        and stats.dollar_volume_today >= _EQUITY_MIN_DOLLAR_VOLUME
        and _EQUITY_MIN_PRICE <= stats.price <= _EQUITY_MAX_PRICE
        and stats.avg_spread_pct <= _EQUITY_MAX_SPREAD_PCT
        and stats.atr14_pct >= _EQUITY_MIN_ATR_PCT
    )


def _passes_crypto_filters(stats: CandidateStats) -> bool:
    symbol = stats.symbol.upper()
    if not symbol.endswith("-USDT"):
        return False
    base = symbol[: -len("-USDT")]
    if any(tag in base for tag in _CRYPTO_LEVERAGED_TAGS):
        return False
    return (
        stats.time_of_day_rvol >= _CRYPTO_MIN_RVOL
        and stats.dollar_volume_today >= _CRYPTO_MIN_DOLLAR_VOLUME
        and stats.atr14_pct >= _CRYPTO_MIN_ATR_PCT
    )


def screen(candidates: Sequence[CandidateStats]) -> list[Candidate]:
    """Apply Strategy_Spec.md §1 filters, rank by RVOL, cap at 20 per venue.

    Pure function — no I/O. See this module's docstring for which §1
    filters are not yet implemented.

    Raises:
        ValueError: A candidate names a venue this module doesn't know
            how to filter (only "alpaca" and "okx" today).
    """
    for c in candidates:
        if c.venue not in _KNOWN_VENUES:
            raise ValueError(f"unknown venue {c.venue!r} — expected one of {_KNOWN_VENUES}")

    eligible: list[CandidateStats] = []
    for c in candidates:
        passes = _passes_equity_filters(c) if c.venue == _EQUITY_VENUE else _passes_crypto_filters(c)
        if passes:
            eligible.append(c)

    by_venue: dict[str, list[CandidateStats]] = {}
    for c in eligible:
        by_venue.setdefault(c.venue, []).append(c)

    result: list[Candidate] = []
    for venue_candidates in by_venue.values():
        venue_candidates.sort(key=lambda c: c.time_of_day_rvol, reverse=True)
        for c in venue_candidates[:MAX_UNIVERSE_SIZE]:
            result.append(
                Candidate(
                    symbol=c.symbol, venue=c.venue, time_of_day_rvol=c.time_of_day_rvol,
                    dollar_volume_today=c.dollar_volume_today, price=c.price,
                    avg_spread_pct=c.avg_spread_pct, atr14_pct=c.atr14_pct,
                )
            )
    return result


def build_universe(symbols: Sequence[str], venue: str, *, asof: datetime, lookback_sessions: int = 20) -> list[Candidate]:
    """Screen a caller-supplied watchlist for `venue`, as of `asof`.

    This does NOT discover the watchlist itself — full-market symbol
    scanning (finding which several-thousand tickers to even consider)
    is a separate, larger data-engineering problem not solved here.
    `symbols` is whatever the caller already wants evaluated.

    Raises:
        ValueError: `venue` isn't one this module knows how to screen, or
            a symbol's `vt.data.feed.get_bars` call fails.
    """
    if venue not in _KNOWN_VENUES:
        raise ValueError(f"unknown venue {venue!r} — expected one of {_KNOWN_VENUES}")

    from vt.data import feed as data_feed

    stats: list[CandidateStats] = []
    for symbol in symbols:
        today_bars = data_feed.get_bars(symbol, "1m", start=asof.replace(hour=0, minute=0, second=0, microsecond=0), end=asof)
        history_bars = data_feed.get_bars(symbol, "1m", end=asof, limit=lookback_sessions * 390)

        rvol = time_of_day_rvol(today_bars, history_bars, asof=asof, lookback_sessions=lookback_sessions)
        latest = max(today_bars, key=lambda b: b.time) if today_bars else max(history_bars, key=lambda b: b.time)
        dollar_volume_today = sum(b.volume * b.close for b in today_bars)

        quote = data_feed.get_quote(symbol)
        spread_pct = 0.0
        if quote.bid and quote.ask and quote.bid > 0:
            spread_pct = ((quote.ask - quote.bid) / quote.bid) * 100.0

        atr_pct = _atr14_pct(history_bars)

        stats.append(
            CandidateStats(
                symbol=symbol, venue=venue, time_of_day_rvol=rvol,
                dollar_volume_today=dollar_volume_today, price=latest.close,
                avg_spread_pct=spread_pct, atr14_pct=atr_pct,
            )
        )

    return screen(stats)


def _atr14_pct(bars: Sequence[Bar], period: int = 14) -> float:
    """ATR(14) as a percentage of the latest close (Wilder's smoothing).

    A local, minimal ATR — deliberately not shared with M003's fuller
    Indicator Engine (`vt.indicators.engine`), since `03_Modules.md`'s
    build order has M002 land before M003. If the two ATR
    implementations ever diverge, reconcile then; duplicating this one
    small calculation now is cheaper than a cross-module dependency this
    early.
    """
    if len(bars) < 2:
        return 0.0
    ordered = sorted(bars, key=lambda b: b.time)
    true_ranges: list[float] = []
    prev_close = ordered[0].close
    for bar in ordered[1:]:
        tr = max(bar.high - bar.low, abs(bar.high - prev_close), abs(bar.low - prev_close))
        true_ranges.append(tr)
        prev_close = bar.close
    if not true_ranges:
        return 0.0
    window = true_ranges[-period:] if len(true_ranges) >= period else true_ranges
    atr = sum(window) / len(window)
    latest_close = ordered[-1].close
    if latest_close <= 0:
        return 0.0
    return (atr / latest_close) * 100.0
