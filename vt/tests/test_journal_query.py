"""Tests for M008 -- vt.journal.query (windowed trade queries; see
13_Session_Log.md S030). No dedicated T-number; built alongside
T020/T021 the same session, user-requested.

Run with: pytest vt/tests -m unit
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from vt.journal import query

pytestmark = pytest.mark.unit


def _closed_card(
    r: float,
    *,
    ts_closed: str,
    card_id: str | None = None,
    pnl_usd: float | None = None,
) -> dict:
    outcome = {"r_multiple": r, "exit_reason": "target_1"}
    if pnl_usd is not None:
        outcome["pnl_usd"] = pnl_usd
    return {
        "card_id": card_id or f"VT-{ts_closed}-{r}",
        "outcome": outcome,
        "ts_closed": ts_closed,
    }


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


_NOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# last_n_trades
# --------------------------------------------------------------------------- #


def test_last_n_trades_returns_the_most_recent_n_oldest_first() -> None:
    cards = [
        _closed_card(1.0, ts_closed=_iso(_NOW - timedelta(hours=5))),
        _closed_card(2.0, ts_closed=_iso(_NOW - timedelta(hours=4))),
        _closed_card(3.0, ts_closed=_iso(_NOW - timedelta(hours=3))),
        _closed_card(4.0, ts_closed=_iso(_NOW - timedelta(hours=2))),
        _closed_card(5.0, ts_closed=_iso(_NOW - timedelta(hours=1))),
    ]
    result = query.last_n_trades(cards, 3)

    assert [c["outcome"]["r_multiple"] for c in result] == [3.0, 4.0, 5.0]


def test_last_n_trades_is_order_independent_of_input_order() -> None:
    """Input list is deliberately shuffled -- last_n_trades must sort by
    ts_closed itself, not trust the caller's ordering."""
    cards = [
        _closed_card(5.0, ts_closed=_iso(_NOW - timedelta(hours=1))),
        _closed_card(1.0, ts_closed=_iso(_NOW - timedelta(hours=5))),
        _closed_card(3.0, ts_closed=_iso(_NOW - timedelta(hours=3))),
    ]
    result = query.last_n_trades(cards, 2)

    assert [c["outcome"]["r_multiple"] for c in result] == [3.0, 5.0]


def test_last_n_trades_returns_fewer_than_n_when_not_enough_exist() -> None:
    cards = [_closed_card(1.0, ts_closed=_iso(_NOW))]
    result = query.last_n_trades(cards, 10)
    assert len(result) == 1


def test_last_n_trades_rejects_nonpositive_n() -> None:
    with pytest.raises(ValueError):
        query.last_n_trades([_closed_card(1.0, ts_closed=_iso(_NOW))], 0)
    with pytest.raises(ValueError):
        query.last_n_trades([_closed_card(1.0, ts_closed=_iso(_NOW))], -1)


def test_last_n_trades_raises_on_card_missing_ts_closed() -> None:
    """The exact mistake this guards against: passing read_cards() output
    (includes still-open cards) instead of closed_cards()."""
    open_card = {"card_id": "VT-open", "outcome": None}
    with pytest.raises(query.MissingTimestampError):
        query.last_n_trades([open_card], 1)


# --------------------------------------------------------------------------- #
# trades_in_range
# --------------------------------------------------------------------------- #


def test_trades_in_range_is_half_open_start_inclusive_end_exclusive() -> None:
    start = _NOW - timedelta(hours=2)
    end = _NOW
    cards = [
        _closed_card(1.0, ts_closed=_iso(start - timedelta(minutes=1)), card_id="before"),
        _closed_card(2.0, ts_closed=_iso(start), card_id="at-start"),
        _closed_card(3.0, ts_closed=_iso(start + timedelta(minutes=30)), card_id="inside"),
        _closed_card(4.0, ts_closed=_iso(end), card_id="at-end"),
    ]
    result = query.trades_in_range(cards, start, end)

    ids = [c["card_id"] for c in result]
    assert ids == ["at-start", "inside"]


def test_trades_in_range_rejects_naive_datetimes() -> None:
    cards = [_closed_card(1.0, ts_closed=_iso(_NOW))]
    with pytest.raises(ValueError):
        query.trades_in_range(cards, datetime(2026, 9, 25), datetime(2026, 9, 26))


def test_trades_in_range_rejects_end_before_or_equal_start() -> None:
    cards = [_closed_card(1.0, ts_closed=_iso(_NOW))]
    with pytest.raises(ValueError):
        query.trades_in_range(cards, _NOW, _NOW - timedelta(hours=1))
    with pytest.raises(ValueError):
        query.trades_in_range(cards, _NOW, _NOW)


# --------------------------------------------------------------------------- #
# trades_today / trades_last_hours -- convenience wrappers over trades_in_range
# --------------------------------------------------------------------------- #


def test_trades_today_includes_only_the_current_calendar_day_in_utc() -> None:
    today_start = datetime(2026, 9, 25, tzinfo=timezone.utc)
    cards = [
        _closed_card(1.0, ts_closed=_iso(today_start - timedelta(minutes=1)), card_id="yesterday"),
        _closed_card(2.0, ts_closed=_iso(today_start + timedelta(hours=1)), card_id="today-morning"),
        _closed_card(3.0, ts_closed=_iso(today_start + timedelta(hours=23, minutes=59)), card_id="today-late"),
        _closed_card(4.0, ts_closed=_iso(today_start + timedelta(days=1)), card_id="tomorrow"),
    ]
    result = query.trades_today(cards, now=today_start + timedelta(hours=12))

    ids = [c["card_id"] for c in result]
    assert ids == ["today-morning", "today-late"]


def test_trades_last_hours_windows_a_rolling_lookback() -> None:
    cards = [
        _closed_card(1.0, ts_closed=_iso(_NOW - timedelta(hours=25)), card_id="too-old"),
        _closed_card(2.0, ts_closed=_iso(_NOW - timedelta(hours=23)), card_id="in-window"),
        _closed_card(3.0, ts_closed=_iso(_NOW - timedelta(minutes=5)), card_id="recent"),
    ]
    result = query.trades_last_hours(cards, 24, now=_NOW)

    ids = [c["card_id"] for c in result]
    assert ids == ["in-window", "recent"]


# --------------------------------------------------------------------------- #
# summarize -- windowed R total, optional dollar total, full scorecard
# --------------------------------------------------------------------------- #


def test_summarize_reports_total_r_and_n() -> None:
    cards = [_closed_card(1.0, ts_closed=_iso(_NOW)), _closed_card(-0.5, ts_closed=_iso(_NOW)), _closed_card(2.0, ts_closed=_iso(_NOW))]
    result = query.summarize(cards)

    assert result.n == 3
    assert result.total_r == pytest.approx(2.5)


def test_summarize_total_pnl_usd_is_none_when_no_card_carries_it() -> None:
    cards = [_closed_card(1.0, ts_closed=_iso(_NOW))]
    result = query.summarize(cards)
    assert result.total_pnl_usd is None


def test_summarize_total_pnl_usd_sums_only_cards_that_carry_it() -> None:
    """Mixed window -- some cards have pnl_usd (e.g. real fills), some
    don't (e.g. imported/legacy rows) -- sum must only count what's
    actually there, and still report a real number, not None, since at
    least one card qualifies."""
    cards = [
        _closed_card(1.0, ts_closed=_iso(_NOW), pnl_usd=12.50),
        _closed_card(-0.5, ts_closed=_iso(_NOW), pnl_usd=-3.25),
        _closed_card(2.0, ts_closed=_iso(_NOW)),  # no pnl_usd
    ]
    result = query.summarize(cards)
    assert result.total_pnl_usd == pytest.approx(9.25)


def test_summarize_scorecard_reflects_the_windowed_n_not_a_fixed_bar() -> None:
    """A 3-trade window is honestly below MIN_N_FOR_EXPECTANCY (100) --
    the scorecard must show insufficient_n, not silently pass/fail on a
    sample this small (T027's own guard, exercised through this new
    entry point)."""
    cards = [_closed_card(1.0, ts_closed=_iso(_NOW)), _closed_card(1.0, ts_closed=_iso(_NOW)), _closed_card(1.0, ts_closed=_iso(_NOW))]
    result = query.summarize(cards)

    assert result.scorecard.n == 3
    assert result.scorecard.honest is False
    assert result.scorecard.get("expectancy_r").verdict == "insufficient_n"


def test_summarize_honors_a_caller_supplied_n_bar() -> None:
    cards = [_closed_card(1.0, ts_closed=_iso(_NOW)) for _ in range(5)]
    result = query.summarize(cards, n_bar=5)

    assert result.scorecard.honest is True
    assert result.scorecard.get("expectancy_r").verdict == "pass"


def test_summarize_composes_cleanly_with_last_n_trades() -> None:
    """The intended real usage: window first, then summarize."""
    cards = [_closed_card(float(i), ts_closed=_iso(_NOW - timedelta(hours=10 - i))) for i in range(10)]
    windowed = query.last_n_trades(cards, 3)
    result = query.summarize(windowed)

    assert result.n == 3
    assert result.total_r == pytest.approx(7.0 + 8.0 + 9.0)
