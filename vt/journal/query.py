"""M008 -- Time/count-windowed trade queries. Phase E follow-up
(user-requested, S030: "profit/losses for a given trade item in a
given time range... past 10 trades, past 24 hours, current day").

`vt.journal.metrics.scorecard`'s own docstring already says the caller
is expected to window the input before calling it ("this function
doesn't slice") -- this module is that windowing layer, the piece that
was never built. It doesn't change what "good" means: everything here
still feeds `closed_cards()` output into the same `scorecard()` /
`cumulative_r()` from `vt.journal.metrics`, which stay the single
source of truth.

R, not dollars, stays the primary unit -- `metrics.py`'s own module
docstring states why: "dollar P&L conflates 'was this a good decision'
with 'how big was the bet' -- kept separate so the history is
teachable." This module doesn't relitigate that. It DOES also report a
dollar total when available, because "how much did I actually make"
is a fair question to want answered too, alongside (not instead of) R
-- `patch_outcome` already forwards any extra outcome field verbatim,
so a caller that wants dollar P&L tracked can add an optional
`pnl_usd` field to the outcome mapping; `summarize()` below sums it
when present and reports `None` (not 0.0) when no card in the window
carries one, so "no data" is never confused with "broke even."

Full contract in `03_Modules.md` section M008; no dedicated T-number
(built alongside T021/T020 in the same session, not part of the
original M010 test spec).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from vt.journal.metrics import MIN_N_FOR_EXPECTANCY, Scorecard, cumulative_r, scorecard


class MissingTimestampError(ValueError):
    """Raised when a card has no `ts_closed` -- the caller passed
    `read_cards()` output (which includes still-open cards) instead of
    `closed_cards()`. Windowing by time is meaningless for a trade that
    hasn't closed yet.
    """


def _ts_closed(card: Mapping[str, Any]) -> datetime:
    raw = card.get("ts_closed")
    if not raw:
        raise MissingTimestampError(
            f"card {card.get('card_id')!r} has no ts_closed -- pass closed_cards() "
            "output (or equivalent), not read_cards(), which includes still-open cards"
        )
    return datetime.fromisoformat(raw)


def last_n_trades(cards: Sequence[Mapping[str, Any]], n: int) -> list[Mapping[str, Any]]:
    """The most recent `n` closed trades by `ts_closed`, oldest first --
    matches `scorecard`/`metrics.max_drawdown_r`'s expected chronological
    input order. Returns fewer than `n` if fewer exist; never raises for
    that, only for a nonpositive `n` or a card missing `ts_closed`.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    return sorted(cards, key=_ts_closed)[-n:]


def trades_in_range(cards: Sequence[Mapping[str, Any]], start: datetime, end: datetime) -> list[Mapping[str, Any]]:
    """Closed trades with `start <= ts_closed < end`, oldest first.
    Half-open so adjacent windows (e.g. yesterday/today) never double-count
    a trade that closed exactly on the boundary.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start/end must be timezone-aware")
    if end <= start:
        raise ValueError(f"end ({end}) must be after start ({start})")
    return sorted((c for c in cards if start <= _ts_closed(c) < end), key=_ts_closed)


def trades_today(
    cards: Sequence[Mapping[str, Any]], *, now: datetime | None = None, tz: timezone = timezone.utc
) -> list[Mapping[str, Any]]:
    """Convenience wrapper: the current calendar day in `tz` (default
    UTC). Pass a different `tz` (e.g. US/Eastern) if "today" should mean
    the trading day rather than the UTC day.
    """
    now = (now or datetime.now(tz)).astimezone(tz)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return trades_in_range(cards, start, start + timedelta(days=1))


def trades_last_hours(
    cards: Sequence[Mapping[str, Any]], hours: float, *, now: datetime | None = None
) -> list[Mapping[str, Any]]:
    """Convenience wrapper: a rolling lookback window, e.g. `trades_last_
    hours(cards, 24)` for "the past 24 hours" (a fixed 24h window, as
    opposed to `trades_today`'s calendar-day boundary)."""
    now = now or datetime.now(timezone.utc)
    return trades_in_range(cards, now - timedelta(hours=hours), now)


@dataclass(frozen=True)
class WindowSummary:
    """`total_pnl_usd` is `None` (not `0.0`) when nothing in the window
    carries the optional `pnl_usd` outcome field -- "no dollar data" and
    "broke even" must never look the same.
    """

    n: int
    total_r: float
    total_pnl_usd: float | None
    scorecard: Scorecard


def _total_pnl_usd(cards: Sequence[Mapping[str, Any]]) -> float | None:
    values = [
        c["outcome"]["pnl_usd"]
        for c in cards
        if isinstance(c.get("outcome"), Mapping) and "pnl_usd" in c["outcome"]
    ]
    return float(sum(values)) if values else None


def summarize(cards: Sequence[Mapping[str, Any]], *, n_bar: int = MIN_N_FOR_EXPECTANCY) -> WindowSummary:
    """Summarize an already-windowed list of closed cards (from
    `last_n_trades` / `trades_in_range` / `trades_today` /
    `trades_last_hours`) -- total R, an optional dollar total, and the
    full `scorecard()` for the window. `n_bar` defaults to `MIN_N_
    FOR_EXPECTANCY` (100): a 10-trade or 24-hour window will correctly
    show every gated verdict as `insufficient_n`, per the T027 guard --
    that's honest, not broken. Pass a smaller `n_bar` explicitly if a
    verdict on a smaller sample is genuinely wanted.
    """
    cards = list(cards)
    return WindowSummary(
        n=len(cards),
        total_r=cumulative_r(cards),
        total_pnl_usd=_total_pnl_usd(cards),
        scorecard=scorecard(cards, n_bar=n_bar),
    )
