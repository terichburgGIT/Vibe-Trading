"""Tests for M001 — vt.data.feed (T001, T002; see 06_Tests.md).

Unit only: the connector calls (`alpaca.sdk.get_historical_bars`,
`okx.sdk.get_historical_bars`, etc.) are monkeypatched to return fixtures
shaped exactly like the real connectors' return payloads, so these tests
never touch the network or need live credentials. Run with:

    pytest vt/tests -m unit

(bare `pytest` from the fork root only discovers upstream's own
`agent/tests` per its `pyproject.toml` — pass this path explicitly.)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from vt.data import feed

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fixtures shaped like the real connector responses
# --------------------------------------------------------------------------- #

_ALPACA_BARS_OK = {
    "status": "ok",
    "symbol": "AAPL",
    "period": "1d",
    "bars": [
        {"time": "2026-08-25T00:00:00Z", "open": 190.0, "high": 192.5, "low": 189.0, "close": 191.0, "volume": 1_000_000},
        {"time": "2026-08-26T00:00:00Z", "open": 191.0, "high": 193.0, "low": 190.5, "close": 192.5, "volume": 1_100_000},
        {"time": "2026-08-24T00:00:00Z", "open": 188.0, "high": 190.5, "low": 187.5, "close": 190.0, "volume": 900_000},
    ],
}

_OKX_BARS_OK = {
    "status": "ok",
    "profile": "paper",
    "is_demo": True,
    "symbol": "BTC-USDT",
    "period": "1d",
    "bar": "1D",
    "bars": [
        # OKX candle dicts carry epoch-millisecond string timestamps.
        {"time": "1756080000000", "open": 60000.0, "high": 61000.0, "low": 59500.0, "close": 60800.0, "volume": 500.0},
        {"time": "1756166400000", "open": 60800.0, "high": 62000.0, "low": 60500.0, "close": 61500.0, "volume": 480.0},
    ],
}

_ALPACA_QUOTE_OK = {
    "status": "ok",
    "symbol": "AAPL",
    "quote": {"bid": 191.20, "ask": 191.25, "bid_size": 3, "ask_size": 5, "time": "2026-08-27T20:00:00Z"},
}

_OKX_QUOTE_OK = {
    "status": "ok",
    "is_demo": True,
    "symbol": "BTC-USDT",
    "quote": {
        "last": 61200.5, "ask": 61201.0, "ask_size": "0.4", "bid": 61199.0, "bid_size": "0.6",
        "open_24h": 60000.0, "high_24h": 62000.0, "low_24h": 59500.0, "volume_24h": 12000.0,
        "time": "1756123200000",
    },
}


def _alpaca_cfg(feed_name: str = "iex") -> SimpleNamespace:
    return SimpleNamespace(feed=feed_name)


# --------------------------------------------------------------------------- #
# T001 — normalized bars, uniform schema, UTC, source_feed tagged
# --------------------------------------------------------------------------- #


def test_get_bars_alpaca_normalizes_schema_and_tags_source_feed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Alpaca bars come back as vt.data.feed.Bar with UTC times and an iex source tag."""
    from src.trading.connectors.alpaca import sdk as alpaca_sdk

    monkeypatch.setattr(alpaca_sdk, "get_historical_bars", lambda *a, **k: _ALPACA_BARS_OK)
    monkeypatch.setattr(alpaca_sdk, "load_config", lambda: _alpaca_cfg("iex"))

    bars = feed.get_bars("AAPL", "1d")

    assert len(bars) == 3
    for bar in bars:
        assert isinstance(bar, feed.Bar)
        assert bar.time.tzinfo is not None
        assert bar.time.utcoffset() == timedelta(0)
        assert bar.source_feed == "alpaca_iex"
        assert bar.symbol == "AAPL"
        assert isinstance(bar.open, float)
        assert isinstance(bar.volume, float)


def test_get_bars_okx_normalizes_schema_and_tags_source_feed(monkeypatch: pytest.MonkeyPatch) -> None:
    """OKX candles (epoch-ms, positional-derived dict) normalize to the same Bar schema."""
    from src.trading.connectors.okx import sdk as okx_sdk

    monkeypatch.setattr(okx_sdk, "get_historical_bars", lambda *a, **k: _OKX_BARS_OK)

    bars = feed.get_bars("BTC-USDT", "1d")

    assert len(bars) == 2
    for bar in bars:
        assert isinstance(bar, feed.Bar)
        assert bar.time.tzinfo is not None
        assert bar.time.utcoffset() == timedelta(0)
        assert bar.source_feed == "okx_demo"
        assert bar.symbol == "BTC-USDT"
        assert isinstance(bar.close, float)


def test_get_bars_schema_is_identical_across_venues(monkeypatch: pytest.MonkeyPatch) -> None:
    """Alpaca and OKX bars expose exactly the same field set (T001's core assertion)."""
    from src.trading.connectors.alpaca import sdk as alpaca_sdk
    from src.trading.connectors.okx import sdk as okx_sdk

    monkeypatch.setattr(alpaca_sdk, "get_historical_bars", lambda *a, **k: _ALPACA_BARS_OK)
    monkeypatch.setattr(alpaca_sdk, "load_config", lambda: _alpaca_cfg("iex"))
    monkeypatch.setattr(okx_sdk, "get_historical_bars", lambda *a, **k: _OKX_BARS_OK)

    equity_bar = feed.get_bars("AAPL", "1d")[0]
    crypto_bar = feed.get_bars("BTC-USDT", "1d")[0]

    assert {f for f in equity_bar.__dataclass_fields__} == {f for f in crypto_bar.__dataclass_fields__}


def test_get_bars_are_chronologically_sorted_with_no_duplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bars come back sorted ascending — the raw Alpaca fixture above is deliberately out of order.

    Full RTH session-gap validation (the other half of T001) needs a trading
    calendar, which doesn't exist yet (M002/M004) — deferred; tracked here so
    the gap isn't silently dropped. See 06_Tests.md T001.
    """
    from src.trading.connectors.alpaca import sdk as alpaca_sdk

    monkeypatch.setattr(alpaca_sdk, "get_historical_bars", lambda *a, **k: _ALPACA_BARS_OK)
    monkeypatch.setattr(alpaca_sdk, "load_config", lambda: _alpaca_cfg("iex"))

    bars = feed.get_bars("AAPL", "1d")

    times = [b.time for b in bars]
    assert times == sorted(times)
    assert len(times) == len(set(times))


def test_get_bars_raises_data_feed_error_on_connector_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A connector error payload must not silently become an empty bar list."""
    from src.trading.connectors.alpaca import sdk as alpaca_sdk

    monkeypatch.setattr(
        alpaca_sdk, "get_historical_bars", lambda *a, **k: {"status": "error", "error": "unauthorized."}
    )
    monkeypatch.setattr(alpaca_sdk, "load_config", lambda: _alpaca_cfg("iex"))

    with pytest.raises(feed.DataFeedError):
        feed.get_bars("AAPL", "1d")


def test_get_quote_alpaca_and_okx_normalize_to_the_same_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.trading.connectors.alpaca import sdk as alpaca_sdk
    from src.trading.connectors.okx import sdk as okx_sdk

    monkeypatch.setattr(alpaca_sdk, "get_quote", lambda *a, **k: _ALPACA_QUOTE_OK)
    monkeypatch.setattr(alpaca_sdk, "load_config", lambda: _alpaca_cfg("iex"))
    monkeypatch.setattr(okx_sdk, "get_quote", lambda *a, **k: _OKX_QUOTE_OK)

    equity_quote = feed.get_quote("AAPL")
    crypto_quote = feed.get_quote("BTC-USDT")

    assert equity_quote.source_feed == "alpaca_iex"
    assert equity_quote.last is None  # Alpaca's latest-quote endpoint has no trade price
    assert crypto_quote.source_feed == "okx_demo"
    assert crypto_quote.last == pytest.approx(61200.5)
    assert {f for f in equity_quote.__dataclass_fields__} == {f for f in crypto_quote.__dataclass_fields__}
    for q in (equity_quote, crypto_quote):
        assert q.time.tzinfo is not None
        assert q.time.utcoffset() == timedelta(0)


# --------------------------------------------------------------------------- #
# T002 — stale quote detection (Risk_Policy.md: quote > 5s old -> reject)
# --------------------------------------------------------------------------- #


def test_is_stale_true_when_quote_is_six_seconds_old() -> None:
    """T002's literal case: a quote 6s old must be flagged stale."""
    quote = feed.Quote(
        symbol="AAPL", bid=100.0, ask=100.05, last=None,
        time=datetime.now(timezone.utc) - timedelta(seconds=6),
        source_feed="alpaca_iex",
    )
    assert feed.is_stale(quote) is True


def test_is_stale_false_when_quote_is_one_second_old() -> None:
    quote = feed.Quote(
        symbol="AAPL", bid=100.0, ask=100.05, last=None,
        time=datetime.now(timezone.utc) - timedelta(seconds=1),
        source_feed="alpaca_iex",
    )
    assert feed.is_stale(quote) is False


def test_is_stale_respects_the_risk_policy_five_second_threshold_exactly() -> None:
    """Risk_Policy.md section 'Data staleness': quote > 5s old -> reject.

    Exact-boundary comparison needs a fixed reference clock, not real
    wall-clock time (two back-to-back `datetime.now()` calls are never
    exactly 5.000000s apart) — `is_stale`'s `now` param exists for this.
    """
    quote_time = datetime.now(timezone.utc) - timedelta(seconds=30)
    quote = feed.Quote(symbol="AAPL", bid=100.0, ask=100.05, last=None, time=quote_time, source_feed="alpaca_iex")

    assert feed.is_stale(quote, now=quote_time + timedelta(seconds=5)) is False
    assert feed.is_stale(quote, now=quote_time + timedelta(seconds=5, microseconds=1)) is True
