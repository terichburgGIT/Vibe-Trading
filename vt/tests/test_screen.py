"""Tests for M002 — vt.universe.screen (T003, T004; see 06_Tests.md).

Unit only, all synthetic — no network, no live connector calls. `screen()`
is a pure function over `CandidateStats`, which is exactly what T003 asks
for ("property test over synthetic candidates"). `build_universe()`, the
I/O-touching orchestrator on top, is intentionally not covered here (it
just wires vt.data.feed bars into CandidateStats and calls screen()).

Run with: pytest vt/tests -m unit
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import pytest

from vt.data.feed import Bar
from vt.universe import screen

pytestmark = pytest.mark.unit


def _bar(day: date, hh: int, mm: int, volume: float, close: float = 100.0, symbol: str = "AAPL") -> Bar:
    return Bar(
        time=datetime(day.year, day.month, day.day, hh, mm, tzinfo=timezone.utc),
        open=close, high=close, low=close, close=close, volume=volume,
        symbol=symbol, source_feed="alpaca_iex",
    )


# --------------------------------------------------------------------------- #
# T004 — RVOL must be time-of-day aware, not compared to a full-day average
# --------------------------------------------------------------------------- #


def test_time_of_day_rvol_matches_the_spec_example_exactly() -> None:
    """06_Tests.md T004, literally: 30% of typical full-day volume by 10:00
    must read as RVOL ~3.0 against a time-matched baseline, not ~0.3 against
    a full-day baseline. This is *the* bug T004 exists to catch.
    """
    today = date(2026, 9, 8)  # a Tuesday, arbitrary
    asof = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)

    # Today: 300,000 shares traded by 10:00 (one bar carrying the cumulative print).
    today_bars = [_bar(today, 9, 30, 300_000.0)]

    # 20 prior sessions: each traded ~100,000 by 10:00, but ~1,000,000 for the
    # full day (the second, later bar) — this full-day total is what a naive
    # (buggy) RVOL would divide by, producing 0.3 instead of 3.0.
    history_bars: list[Bar] = []
    for i in range(1, 21):
        session = date(2026, 8, 31 - i) if (31 - i) > 0 else date(2026, 7, 31 - (i - 31))
        history_bars.append(_bar(session, 9, 30, 100_000.0))
        history_bars.append(_bar(session, 15, 30, 900_000.0))  # rest of the day's volume

    rvol = screen.time_of_day_rvol(today_bars, history_bars, asof=asof)

    assert rvol == pytest.approx(3.0, rel=0.01)

    # The naive (wrong) computation the test name warns against, spelled out
    # so the contrast is explicit rather than just asserted away:
    naive_full_day_baseline = 1_000_000.0
    naive_rvol = 300_000.0 / naive_full_day_baseline
    assert naive_rvol == pytest.approx(0.3, rel=0.01)
    assert rvol != pytest.approx(naive_rvol, rel=0.5)


def test_time_of_day_rvol_ignores_bars_after_the_cutoff() -> None:
    """A bar at 14:00 today must not count toward a 10:00 cutoff RVOL."""
    today = date(2026, 9, 8)
    asof = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)
    today_bars = [_bar(today, 9, 30, 200_000.0), _bar(today, 14, 0, 5_000_000.0)]
    history_bars = [_bar(date(2026, 9, 8 - i), 9, 30, 100_000.0) for i in range(1, 6)]

    rvol = screen.time_of_day_rvol(today_bars, history_bars, asof=asof)

    assert rvol == pytest.approx(2.0, rel=0.01)  # 200k / 100k, the 5M afternoon bar excluded


def test_time_of_day_rvol_raises_with_no_history_sessions() -> None:
    today = date(2026, 9, 8)
    asof = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        screen.time_of_day_rvol([_bar(today, 9, 30, 100.0)], [], asof=asof)


def test_time_of_day_rvol_raises_on_zero_baseline() -> None:
    today = date(2026, 9, 8)
    asof = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)
    history_bars = [_bar(date(2026, 9, 8 - i), 11, 0, 100.0) for i in range(1, 4)]  # all after cutoff
    with pytest.raises(ValueError):
        screen.time_of_day_rvol([_bar(today, 9, 30, 100.0)], history_bars, asof=asof)


# --------------------------------------------------------------------------- #
# T004 (crypto sibling) — RVOL must be hour-of-week aware, not just
# hour-of-day, because crypto trades 24/7 and Saturday-3pm volume is
# structurally different from Tuesday-3pm volume.
# --------------------------------------------------------------------------- #


def _hourly_bar(dt: datetime, volume: float, close: float = 80_000.0, symbol: str = "BTC-USDT") -> Bar:
    return Bar(time=dt, open=close, high=close, low=close, close=close, volume=volume, symbol=symbol, source_feed="okx_demo")


def test_hour_of_week_rvol_ignores_other_weekdays_in_the_same_hour_slot() -> None:
    """2026-09-05 is a Saturday. Baseline must average only prior Saturdays
    at hour 15, never the much-busier Tuesday-15:00 bars mixed into the same
    `history_bars` set — that's the hour-of-week analogue of T004's trap.
    """
    asof = datetime(2026, 9, 5, 15, 30, tzinfo=timezone.utc)  # Saturday, 30m into the 15:00 hour
    today_bars = [_hourly_bar(datetime(2026, 9, 5, 15, 0, tzinfo=timezone.utc), 300.0)]

    # 4 prior Saturdays at hour 15: ~100 volume each (the real baseline).
    saturdays = [datetime(2026, 9, 5, tzinfo=timezone.utc) - timedelta(weeks=w) for w in range(1, 5)]
    history_bars = [_hourly_bar(sat.replace(hour=15), 100.0) for sat in saturdays]
    # Two Tuesdays at hour 15: ~500 volume each — must NOT dilute the baseline.
    history_bars += [
        _hourly_bar(datetime(2026, 9, 1, 15, 0, tzinfo=timezone.utc), 500.0),
        _hourly_bar(datetime(2026, 8, 25, 15, 0, tzinfo=timezone.utc), 500.0),
    ]

    rvol = screen.hour_of_week_rvol(today_bars, history_bars, asof=asof, lookback_weeks=4)

    assert rvol == pytest.approx(3.0, rel=0.01)  # 300 / 100, Tuesdays excluded


def test_hour_of_week_rvol_ignores_bars_outside_the_current_hour_slot() -> None:
    asof = datetime(2026, 9, 5, 15, 30, tzinfo=timezone.utc)
    today_bars = [
        _hourly_bar(datetime(2026, 9, 5, 15, 0, tzinfo=timezone.utc), 200.0),
        _hourly_bar(datetime(2026, 9, 5, 20, 0, tzinfo=timezone.utc), 9_000.0),  # later same day, must not count
    ]
    history_bars = [
        _hourly_bar(datetime(2026, 8, 29, 15, 0, tzinfo=timezone.utc), 100.0),  # prior Saturday
    ]

    rvol = screen.hour_of_week_rvol(today_bars, history_bars, asof=asof, lookback_weeks=4)

    assert rvol == pytest.approx(2.0, rel=0.01)  # 200 / 100, the 9000 bar excluded


def test_hour_of_week_rvol_raises_with_no_matching_occurrences() -> None:
    asof = datetime(2026, 9, 5, 15, 30, tzinfo=timezone.utc)
    today_bars = [_hourly_bar(datetime(2026, 9, 5, 15, 0, tzinfo=timezone.utc), 100.0)]
    # Only Tuesday history — no prior Saturday-15:00 occurrence exists.
    history_bars = [_hourly_bar(datetime(2026, 9, 1, 15, 0, tzinfo=timezone.utc), 500.0)]
    with pytest.raises(ValueError):
        screen.hour_of_week_rvol(today_bars, history_bars, asof=asof)


def test_hour_of_week_rvol_raises_on_zero_baseline() -> None:
    asof = datetime(2026, 9, 5, 15, 30, tzinfo=timezone.utc)
    today_bars = [_hourly_bar(datetime(2026, 9, 5, 15, 0, tzinfo=timezone.utc), 100.0)]
    history_bars = [_hourly_bar(datetime(2026, 8, 29, 15, 0, tzinfo=timezone.utc), 0.0)]
    with pytest.raises(ValueError):
        screen.hour_of_week_rvol(today_bars, history_bars, asof=asof)


# --------------------------------------------------------------------------- #
# T003 — screen() respects every *implemented* Strategy_Spec.md §1 filter
# --------------------------------------------------------------------------- #


def _equity_stats(**overrides: object) -> screen.CandidateStats:
    base = dict(
        symbol="AAPL", venue="alpaca", time_of_day_rvol=3.0,
        dollar_volume_today=25_000_000.0, price=150.0, avg_spread_pct=0.05, atr14_pct=2.0,
    )
    base.update(overrides)
    return screen.CandidateStats(**base)  # type: ignore[arg-type]


def _crypto_stats(**overrides: object) -> screen.CandidateStats:
    base = dict(
        symbol="BTC-USDT", venue="okx", time_of_day_rvol=3.0,
        dollar_volume_today=80_000_000.0, price=61_000.0, avg_spread_pct=0.02, atr14_pct=2.5,
    )
    base.update(overrides)
    return screen.CandidateStats(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    [
        {"time_of_day_rvol": 1.9},           # below 2.0x minimum
        {"dollar_volume_today": 19_999_999},  # below $20M
        {"price": 4.99},                      # below $5 floor
        {"price": 500.01},                    # above $500 ceiling
        {"avg_spread_pct": 0.16},              # above 0.15% max
        {"atr14_pct": 1.49},                  # below 1.5% minimum
    ],
)
def test_screen_rejects_equities_violating_any_single_threshold(overrides: dict) -> None:
    stats = _equity_stats(**overrides)
    assert screen.screen([stats]) == []


def test_screen_accepts_an_equity_candidate_meeting_every_threshold() -> None:
    stats = _equity_stats()
    result = screen.screen([stats])
    assert len(result) == 1
    assert result[0].symbol == "AAPL"


@pytest.mark.parametrize(
    "overrides",
    [
        {"time_of_day_rvol": 1.9},
        {"dollar_volume_today": 49_999_999},
        {"atr14_pct": 1.49},
        {"symbol": "ETH-USD"},        # not USDT-quoted
        {"symbol": "BTCUP-USDT"},     # leveraged token
        {"symbol": "BTCBEAR-USDT"},   # leveraged token
    ],
)
def test_screen_rejects_crypto_violating_any_single_threshold(overrides: dict) -> None:
    stats = _crypto_stats(**overrides)
    assert screen.screen([stats]) == []


def test_screen_accepts_a_crypto_candidate_meeting_every_threshold() -> None:
    result = screen.screen([_crypto_stats()])
    assert len(result) == 1
    assert result[0].symbol == "BTC-USDT"


def test_screen_caps_each_venue_at_twenty_names_ranked_by_rvol() -> None:
    """Strategy_Spec.md §1: 'Produces <=20 names per venue.'"""
    many = [_equity_stats(symbol=f"SYM{i}", time_of_day_rvol=2.0 + i * 0.1) for i in range(30)]
    result = screen.screen(many)
    assert len(result) == screen.MAX_UNIVERSE_SIZE
    # Highest RVOL kept, not an arbitrary/first-N slice.
    assert result[0].symbol == "SYM29"
    assert result[0].time_of_day_rvol == pytest.approx(2.0 + 29 * 0.1)


def test_screen_caps_are_independent_per_venue() -> None:
    equities = [_equity_stats(symbol=f"SYM{i}", time_of_day_rvol=2.0 + i * 0.1) for i in range(25)]
    crypto = [_crypto_stats(symbol=f"COIN{i}-USDT", time_of_day_rvol=2.0 + i * 0.1) for i in range(25)]
    result = screen.screen(equities + crypto)
    by_venue: dict[str, int] = {}
    for c in result:
        by_venue[c.venue] = by_venue.get(c.venue, 0) + 1
    assert by_venue == {"alpaca": screen.MAX_UNIVERSE_SIZE, "okx": screen.MAX_UNIVERSE_SIZE}


def test_screen_rejects_unknown_venue() -> None:
    stats = _equity_stats(venue="robinhood")
    with pytest.raises(ValueError):
        screen.screen([stats])


# --------------------------------------------------------------------------- #
# _atr14_pct — golden-value check (not exercised by the CandidateStats tests
# above, which hardcode atr14_pct directly rather than deriving it from bars)
# --------------------------------------------------------------------------- #


def test_atr14_pct_matches_a_hand_computed_golden_value() -> None:
    """Four bars, constant true range of 2.0 each step, latest close 12.0.

    True ranges: max(high-low, |high-prev_close|, |low-prev_close|) for each
    bar after the seed = [2, 2, 2] (hand-computed). Only 3 true ranges exist,
    so the window is all 3 (fewer than the 14-period default), giving
    atr = 2.0 exactly. atr_pct = 2.0 / 12.0 * 100 = 16.666...%.
    """
    day = date(2026, 9, 8)
    bars = [
        _bar(day, 9, 30, 1000.0, close=9.0),   # seed bar, only its close is used
        _bar(day, 9, 31, 1000.0, close=10.0),
        _bar(day, 9, 32, 1000.0, close=11.0),
        _bar(day, 9, 33, 1000.0, close=12.0),
    ]
    # _bar() sets high=low=close, which would zero out true range — build
    # real high/low directly instead for this test.
    from dataclasses import replace
    bars = [
        replace(bars[0], high=10.0, low=8.0),
        replace(bars[1], high=11.0, low=9.0),
        replace(bars[2], high=12.0, low=10.0),
        replace(bars[3], high=13.0, low=11.0),
    ]

    atr_pct = screen._atr14_pct(bars)

    assert atr_pct == pytest.approx(2.0 / 12.0 * 100.0, rel=0.001)


def test_atr14_pct_is_zero_with_fewer_than_two_bars() -> None:
    assert screen._atr14_pct([]) == 0.0
    assert screen._atr14_pct([_bar(date(2026, 9, 8), 9, 30, 100.0)]) == 0.0
