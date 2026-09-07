"""M001 — Data Layer.

Unified OHLCV + quote access across Alpaca (equities) and OKX (crypto),
normalized to one bar/quote schema with UTC timestamps and a
`source_feed` tag (AD003 — IEX vs. SIP vs. demo must stay distinguishable
downstream). Thin wrapper over upstream connectors
(`src.trading.connectors.*`) — see AD001; this file never edits them.

Venue routing is symbol-shape based: a symbol containing "-" (e.g.
"BTC-USDT") routes to OKX, everything else routes to Alpaca. Kraken and
yfinance are not wired in yet (Kraken is parked per AD014; yfinance
fallback is a follow-up once a gap in Alpaca/OKX coverage actually shows
up — no upstream loader call has been added speculatively).

Full contract: `03_Modules.md` § M001. Tests: `vt/tests/test_feed.py`
(T001, T002 in `06_Tests.md`).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timedelta, timezone
from typing import Any

_EQUITY_VENUE = "alpaca"
_CRYPTO_VENUE = "okx"

# Risk_Policy.md "Data staleness": quote > 5s old -> reject all new orders.
_DEFAULT_MAX_QUOTE_AGE_SECONDS = 5.0


class DataFeedError(RuntimeError):
    """Raised when a venue connector returns a non-ok status.

    Deliberately not swallowed into an empty result — a caller silently
    trading on an empty bar list is worse than a loud failure.
    """


@dataclass(frozen=True)
class Bar:
    """One OHLCV bar, normalized across every venue this module supports."""

    time: datetime  # tz-aware, UTC
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    source_feed: str  # e.g. "alpaca_iex", "okx_demo", "okx_live"


@dataclass(frozen=True)
class Quote:
    """One top-of-book quote snapshot, normalized across every venue."""

    symbol: str
    bid: float | None
    ask: float | None
    last: float | None  # Alpaca's latest-quote endpoint has no trade price -> None
    time: datetime  # tz-aware, UTC
    source_feed: str


def _venue_for_symbol(symbol: str) -> str:
    """Crypto pairs carry a dash (BTC-USDT); equities are bare tickers (AAPL)."""
    return _CRYPTO_VENUE if "-" in symbol else _EQUITY_VENUE


def _default_alpaca_start(timeframe: str, limit: int) -> datetime:
    """A generous lookback window sized so Alpaca actually returns `limit` bars.

    Alpaca's bars endpoint returns an empty result when `start` is omitted
    entirely (AD001 exception 1 — confirmed live 2026-09-07) rather than
    defaulting to "most recent N bars". `limit` counts trading periods, not
    calendar ones, so this pads well past the raw calendar equivalent to
    absorb weekends and holidays.
    """
    unit = timeframe[-1]
    try:
        amount = int(timeframe[:-1])
    except ValueError:
        amount = 1
    if unit == "d":
        calendar_days = limit * amount * 3  # ~3x pads weekends/holidays
    elif unit == "h":
        calendar_days = (limit * amount / 6.5) * 3  # ~6.5 trading hours/day
    else:  # minutes or an unrecognized unit — pad generously either way
        calendar_days = (limit * amount / 390) * 3  # ~390 trading minutes/day
    return datetime.now(timezone.utc) - timedelta(days=max(calendar_days, 7))


def _maybe_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _parse_alpaca_time(raw: str) -> datetime:
    """Alpaca timestamps are ISO-8601 strings, sometimes with a trailing 'Z'."""
    if not raw:
        raise ValueError("empty timestamp from Alpaca")
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_okx_time(raw: str) -> datetime:
    """OKX timestamps are epoch milliseconds, delivered as a string."""
    if not raw:
        raise ValueError("empty timestamp from OKX")
    return datetime.fromtimestamp(int(raw) / 1000, tz=timezone.utc)


def _normalize_alpaca_bar(raw: dict[str, Any], symbol: str, source_feed: str) -> Bar:
    return Bar(
        time=_parse_alpaca_time(raw["time"]),
        open=float(raw["open"]),
        high=float(raw["high"]),
        low=float(raw["low"]),
        close=float(raw["close"]),
        volume=float(raw["volume"]),
        symbol=symbol,
        source_feed=source_feed,
    )


def _normalize_okx_bar(raw: dict[str, Any], symbol: str, source_feed: str) -> Bar:
    return Bar(
        time=_parse_okx_time(raw["time"]),
        open=float(raw["open"]),
        high=float(raw["high"]),
        low=float(raw["low"]),
        close=float(raw["close"]),
        volume=float(raw["volume"]),
        symbol=symbol,
        source_feed=source_feed,
    )


def get_bars(
    symbol: str,
    timeframe: str = "1d",
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = 90,
) -> list[Bar]:
    """Fetch normalized, UTC-sorted bars for `symbol` from its venue.

    Args:
        symbol: Bare ticker for equities ("AAPL") or a dashed pair for
            crypto ("BTC-USDT") — the dash is what selects the venue.
        timeframe: Canonical period token (e.g. "1d", "1h") — passed
            through to the connector, which maps it to that venue's own
            timeframe format.
        start: Optional inclusive UTC lower bound. Passed through to the
            connector (both Alpaca and OKX page/paginate against it as of
            AD001 exceptions 1 and 2), then re-applied client-side as a
            belt-and-suspenders filter. Alpaca omitting `start` entirely
            returns zero bars (not "most recent N"), so when the caller
            doesn't supply one, Alpaca calls compute a generous default
            (`_default_alpaca_start`) — OKX has no such requirement.
        end: Optional inclusive UTC upper bound, applied the same way.
        limit: Bars requested from the connector before any start/end
            filtering. For OKX, values above 300 trigger multi-page
            pagination inside the connector (AD001 exception 2).

    Returns:
        Bars sorted ascending by time, deduplicated by construction (each
        connector returns one row per period), every bar carrying its
        `source_feed` tag.

    Raises:
        DataFeedError: The connector returned a non-ok status.
    """
    venue = _venue_for_symbol(symbol)
    if venue == _EQUITY_VENUE:
        from src.trading.connectors.alpaca import sdk as alpaca_sdk

        alpaca_start = start if start is not None else _default_alpaca_start(timeframe, limit)
        raw = alpaca_sdk.get_historical_bars(symbol, period=timeframe, limit=limit, start=alpaca_start, end=end)
        if raw.get("status") != "ok":
            raise DataFeedError(f"alpaca get_historical_bars failed: {raw.get('error')}")
        source_feed = f"alpaca_{alpaca_sdk.load_config().feed}"
        bars = [_normalize_alpaca_bar(row, symbol, source_feed) for row in raw["bars"]]
    else:
        from src.trading.connectors.okx import sdk as okx_sdk

        raw = okx_sdk.get_historical_bars(symbol, period=timeframe, limit=limit, start=start, end=end)
        if raw.get("status") != "ok":
            raise DataFeedError(f"okx get_historical_bars failed: {raw.get('error')}")
        source_feed = "okx_demo" if raw.get("is_demo") else "okx_live"
        bars = [_normalize_okx_bar(row, symbol, source_feed) for row in raw["bars"]]

    bars.sort(key=lambda b: b.time)
    if start is not None:
        bars = [b for b in bars if b.time >= start]
    if end is not None:
        bars = [b for b in bars if b.time <= end]
    return bars


def get_quote(symbol: str) -> Quote:
    """Fetch a normalized top-of-book quote for `symbol` from its venue.

    Raises:
        DataFeedError: The connector returned a non-ok status.
    """
    venue = _venue_for_symbol(symbol)
    if venue == _EQUITY_VENUE:
        from src.trading.connectors.alpaca import sdk as alpaca_sdk

        raw = alpaca_sdk.get_quote(symbol)
        if raw.get("status") != "ok":
            raise DataFeedError(f"alpaca get_quote failed: {raw.get('error')}")
        q = raw["quote"]
        return Quote(
            symbol=symbol,
            bid=_maybe_float(q.get("bid")),
            ask=_maybe_float(q.get("ask")),
            last=None,
            time=_parse_alpaca_time(q["time"]),
            source_feed=f"alpaca_{alpaca_sdk.load_config().feed}",
        )
    else:
        from src.trading.connectors.okx import sdk as okx_sdk

        raw = okx_sdk.get_quote(symbol)
        if raw.get("status") != "ok":
            raise DataFeedError(f"okx get_quote failed: {raw.get('error')}")
        q = raw["quote"]
        return Quote(
            symbol=symbol,
            bid=_maybe_float(q.get("bid")),
            ask=_maybe_float(q.get("ask")),
            last=_maybe_float(q.get("last")),
            time=_parse_okx_time(q["time"]),
            source_feed="okx_demo" if raw.get("is_demo") else "okx_live",
        )


def is_stale(
    quote: Quote,
    *,
    max_age_seconds: float = _DEFAULT_MAX_QUOTE_AGE_SECONDS,
    now: datetime | None = None,
) -> bool:
    """Whether `quote` is older than `max_age_seconds` (Risk_Policy.md: >5s -> reject).

    `now` is injectable so callers (and tests) can check staleness against
    a fixed reference instant instead of the real wall clock.
    """
    reference = now if now is not None else datetime.now(timezone.utc)
    age = (reference - quote.time).total_seconds()
    return age > max_age_seconds


# Re-exported so callers can introspect the canonical schema without
# reaching into dataclass internals directly.
BAR_FIELDS = tuple(f.name for f in fields(Bar))
QUOTE_FIELDS = tuple(f.name for f in fields(Quote))
