"""Tests for the concrete OKX *spot* execution adapter (vt.exec.okx),
which fills in M007's `BrokerExecAdapter` Protocol against OKX's demo
(or live) spot trading API.

OKX's upstream connector is thinner than Alpaca's, so this adapter wraps
upstream where it can and goes around it only for the one gap upstream
can't express -- exactly the Alpaca shape, just with different seams:

  * Upstream's `src.trading.connectors.okx.sdk` functions
    (`place_order`, `cancel_order`, `get_open_orders`,
    `get_account_snapshot`, `load_config`) are monkeypatched directly on
    the module object -- same pattern `test_feed.py`/`test_exec_alpaca.py`
    use.
  * Raw python-okx (`okx.Trade.TradeAPI`) is faked via `sys.modules`
    injection for the STOP leg only -- upstream has no stop-order type,
    so `submit_stop` calls `place_algo_order` directly, the same
    lazy-SDK-import pattern `vt/alerts/kill.py` and the Alpaca adapter
    already use.

Two OKX-specific facts drive the shape of these tests, both documented
in `vt/exec/okx.py`:
  * OKX spot has no native "position" -- a filled buy just raises the
    base-currency balance. `positions()` therefore synthesizes positions
    from the account balance, reporting each non-quote holding as
    `{ccy}-{quote}` (the quote currency itself is cash, never a
    position).
  * OKX's `cancel` requires the instrument id, which the Protocol's
    `cancel(order_id)` doesn't carry -- so the adapter looks the symbol
    up from the open-orders list first.

Neither strategy touches a real network call or a real OKX account.

Run with: pytest vt/tests -m unit
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from src.trading.connectors.okx import sdk as okx_sdk
from vt.exec import okx as vt_okx

pytestmark = pytest.mark.unit


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        api_key="okx-test-key",
        api_secret="okx-test-secret",
        passphrase="okx-test-pass",
        profile="paper",
        host="https://www.okx.com",
        flag="1",
        is_demo=True,
    )


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_venue_is_okx() -> None:
    adapter = vt_okx.OKXExecAdapter(config=_cfg())
    assert adapter.venue == "okx"


def test_constructing_without_config_loads_from_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(okx_sdk, "load_config", lambda: _cfg())
    adapter = vt_okx.OKXExecAdapter()
    assert adapter._config.api_key == "okx-test-key"


def test_constructing_with_explicit_config_does_not_touch_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom():
        raise AssertionError("load_config should not be called when config is supplied")

    monkeypatch.setattr(okx_sdk, "load_config", _boom)
    adapter = vt_okx.OKXExecAdapter(config=_cfg())
    assert adapter._config.api_key == "okx-test-key"


def test_quote_currency_defaults_to_usdt_and_is_overridable() -> None:
    assert vt_okx.OKXExecAdapter(config=_cfg())._quote_ccy == "USDT"
    assert vt_okx.OKXExecAdapter(config=_cfg(), quote_ccy="usdc")._quote_ccy == "USDC"


# --------------------------------------------------------------------------- #
# submit_entry -- wraps upstream place_order (buy limit, tdMode cash)
# --------------------------------------------------------------------------- #


def test_submit_entry_long_places_a_buy_limit_order(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_place_order(config, **kwargs):
        captured.update(kwargs)
        captured["config"] = config
        return {"status": "ok", "order_id": "entry-123"}

    monkeypatch.setattr(okx_sdk, "place_order", fake_place_order)
    adapter = vt_okx.OKXExecAdapter(config=_cfg())

    order_id = adapter.submit_entry(
        symbol="BTC-USDT", side="long", size=0.01, limit_price=50000.0, client_order_id="coid-1"
    )

    assert order_id == "entry-123"
    assert captured["symbol"] == "BTC-USDT"
    assert captured["side"] == "buy"
    assert captured["quantity"] == 0.01
    assert captured["order_type"] == "limit"
    assert captured["limit_price"] == 50000.0


def test_submit_entry_rejects_short_side_spot_is_long_only(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(config, **kwargs):
        raise AssertionError("place_order must not be reached for a short on spot")

    monkeypatch.setattr(okx_sdk, "place_order", _boom)
    adapter = vt_okx.OKXExecAdapter(config=_cfg())

    with pytest.raises(vt_okx.OKXAdapterError, match="long-only"):
        adapter.submit_entry(
            symbol="BTC-USDT", side="short", size=0.01, limit_price=50000.0, client_order_id="coid-2"
        )


def test_submit_entry_raises_okx_adapter_error_on_broker_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        okx_sdk, "place_order", lambda config, **kwargs: {"status": "error", "error": "Insufficient balance"}
    )
    adapter = vt_okx.OKXExecAdapter(config=_cfg())

    with pytest.raises(vt_okx.OKXAdapterError, match="Insufficient balance"):
        adapter.submit_entry(
            symbol="BTC-USDT", side="long", size=0.01, limit_price=50000.0, client_order_id="coid-3"
        )


# --------------------------------------------------------------------------- #
# submit_stop -- raw python-okx place_algo_order, faked via sys.modules
# --------------------------------------------------------------------------- #


def _install_fake_okx_trade_sdk(monkeypatch: pytest.MonkeyPatch, *, captured: dict[str, Any]) -> None:
    """Install a fake `okx.Trade` module so `submit_stop`'s lazy import of
    `TradeAPI` resolves to a test double instead of the real python-okx
    package."""

    class FakeTradeAPI:
        def __init__(self, api_key, api_secret, passphrase, use_server_time, flag, domain=None) -> None:
            captured["ctor"] = {
                "api_key": api_key,
                "api_secret": api_secret,
                "passphrase": passphrase,
                "flag": flag,
                "domain": domain,
            }

        def place_algo_order(self, **kwargs):
            captured["algo_kwargs"] = kwargs
            if captured.get("fail_submit"):
                raise RuntimeError("algo submit refused by broker")
            if captured.get("business_reject"):
                return {"code": "1", "data": [{"sCode": "51000", "sMsg": "trigger price invalid"}]}
            return {"code": "0", "data": [{"algoId": "stop-789", "algoClOrdId": kwargs.get("algoClOrdId", ""), "sCode": "0", "sMsg": ""}]}

    fake_okx = types.ModuleType("okx")
    fake_trade_mod = types.ModuleType("okx.Trade")
    fake_trade_mod.TradeAPI = FakeTradeAPI
    fake_okx.Trade = fake_trade_mod

    monkeypatch.setitem(sys.modules, "okx", fake_okx)
    monkeypatch.setitem(sys.modules, "okx.Trade", fake_trade_mod)


def test_submit_stop_for_a_long_position_submits_a_conditional_sell(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _install_fake_okx_trade_sdk(monkeypatch, captured=captured)
    adapter = vt_okx.OKXExecAdapter(config=_cfg())

    order_id = adapter.submit_stop(
        symbol="BTC-USDT", side="long", size=0.01, stop_price=48000.0, client_order_id="coidstop"
    )

    assert order_id == "stop-789"
    assert captured["ctor"]["api_key"] == "okx-test-key"
    assert captured["ctor"]["flag"] == "1"
    kwargs = captured["algo_kwargs"]
    assert kwargs["instId"] == "BTC-USDT"
    assert kwargs["tdMode"] == "cash"
    assert kwargs["side"] == "sell"  # long spot -> closing side is sell
    assert kwargs["ordType"] == "conditional"
    assert kwargs["sz"] == "0.01"
    assert kwargs["slTriggerPx"] == "48000.0"
    assert kwargs["slOrdPx"] == "-1"  # market execution when triggered


def test_submit_stop_rejects_short_side_spot_is_long_only(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _install_fake_okx_trade_sdk(monkeypatch, captured=captured)
    adapter = vt_okx.OKXExecAdapter(config=_cfg())

    with pytest.raises(vt_okx.OKXAdapterError, match="long-only"):
        adapter.submit_stop(
            symbol="BTC-USDT", side="short", size=0.01, stop_price=52000.0, client_order_id="coidstop"
        )
    assert "algo_kwargs" not in captured


def test_submit_stop_raises_on_transport_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {"fail_submit": True}
    _install_fake_okx_trade_sdk(monkeypatch, captured=captured)
    adapter = vt_okx.OKXExecAdapter(config=_cfg())

    with pytest.raises(vt_okx.OKXAdapterError, match="algo submit refused"):
        adapter.submit_stop(
            symbol="BTC-USDT", side="long", size=0.01, stop_price=48000.0, client_order_id="coidstop"
        )


def test_submit_stop_raises_on_business_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {"business_reject": True}
    _install_fake_okx_trade_sdk(monkeypatch, captured=captured)
    adapter = vt_okx.OKXExecAdapter(config=_cfg())

    with pytest.raises(vt_okx.OKXAdapterError, match="trigger price invalid"):
        adapter.submit_stop(
            symbol="BTC-USDT", side="long", size=0.01, stop_price=48000.0, client_order_id="coidstop"
        )


def test_submit_stop_forwards_an_okx_valid_client_id_as_algoclordid(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _install_fake_okx_trade_sdk(monkeypatch, captured=captured)
    adapter = vt_okx.OKXExecAdapter(config=_cfg())

    adapter.submit_stop(
        symbol="BTC-USDT", side="long", size=0.01, stop_price=48000.0, client_order_id="alnum32ok"
    )
    assert captured["algo_kwargs"]["algoClOrdId"] == "alnum32ok"


def test_submit_stop_omits_a_hyphenated_client_id_okx_charset(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pipeline derives the stop coid as `<entry_coid>-stop`; OKX's
    clOrdId charset is alphanumeric only, so a hyphenated id can't be
    forwarded. The adapter degrades gracefully -- it omits algoClOrdId
    and lets OKX assign its own id, rather than failing the stop."""
    captured: dict[str, Any] = {}
    _install_fake_okx_trade_sdk(monkeypatch, captured=captured)
    adapter = vt_okx.OKXExecAdapter(config=_cfg())

    order_id = adapter.submit_stop(
        symbol="BTC-USDT", side="long", size=0.01, stop_price=48000.0, client_order_id="coid-1-stop"
    )
    assert order_id == "stop-789"
    assert captured["algo_kwargs"].get("algoClOrdId", "") == ""


# --------------------------------------------------------------------------- #
# cancel -- looks the symbol up from open orders, then wraps upstream
# --------------------------------------------------------------------------- #


def test_cancel_looks_up_symbol_then_delegates_to_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        okx_sdk,
        "get_open_orders",
        lambda config, **_: {
            "status": "ok",
            "open_orders": [
                {"symbol": "ETH-USDT", "order_id": "other-1"},
                {"symbol": "BTC-USDT", "order_id": "entry-123"},
            ],
        },
    )
    captured: dict[str, Any] = {}

    def fake_cancel_order(config, order_id="", *, symbol=None):
        captured["order_id"] = order_id
        captured["symbol"] = symbol
        return {"status": "ok"}

    monkeypatch.setattr(okx_sdk, "cancel_order", fake_cancel_order)
    adapter = vt_okx.OKXExecAdapter(config=_cfg())

    adapter.cancel("entry-123")
    assert captured["order_id"] == "entry-123"
    assert captured["symbol"] == "BTC-USDT"


def test_cancel_raises_when_order_not_among_open_orders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        okx_sdk,
        "get_open_orders",
        lambda config, **_: {"status": "ok", "open_orders": []},
    )

    def _boom(config, order_id="", *, symbol=None):
        raise AssertionError("cancel_order must not be called when the order isn't open")

    monkeypatch.setattr(okx_sdk, "cancel_order", _boom)
    adapter = vt_okx.OKXExecAdapter(config=_cfg())

    with pytest.raises(vt_okx.OKXAdapterError, match="not found among open orders"):
        adapter.cancel("ghost-1")


def test_cancel_raises_on_open_orders_read_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        okx_sdk, "get_open_orders", lambda config, **_: {"status": "error", "error": "auth failed"}
    )
    adapter = vt_okx.OKXExecAdapter(config=_cfg())

    with pytest.raises(vt_okx.OKXAdapterError, match="auth failed"):
        adapter.cancel("entry-123")


def test_cancel_raises_on_broker_cancel_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        okx_sdk,
        "get_open_orders",
        lambda config, **_: {"status": "ok", "open_orders": [{"symbol": "BTC-USDT", "order_id": "entry-123"}]},
    )
    monkeypatch.setattr(
        okx_sdk, "cancel_order", lambda config, order_id="", *, symbol=None: {"status": "error", "error": "order already filled"}
    )
    adapter = vt_okx.OKXExecAdapter(config=_cfg())

    with pytest.raises(vt_okx.OKXAdapterError, match="order already filled"):
        adapter.cancel("entry-123")


# --------------------------------------------------------------------------- #
# close_position -- market-sell the held base balance
# --------------------------------------------------------------------------- #


def _snapshot(details: list[dict[str, Any]]) -> dict[str, Any]:
    return {"status": "ok", "account": {"total_equity": "100000", "details": details}}


def test_close_position_market_sells_the_held_base_balance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        okx_sdk,
        "get_account_snapshot",
        lambda config: _snapshot([{"currency": "BTC", "cash_balance": "0.01", "available": "0.01"}]),
    )
    captured: dict[str, Any] = {}

    def fake_place_order(config, **kwargs):
        captured.update(kwargs)
        return {"status": "ok", "order_id": "flatten-1"}

    monkeypatch.setattr(okx_sdk, "place_order", fake_place_order)
    adapter = vt_okx.OKXExecAdapter(config=_cfg())

    adapter.close_position("BTC-USDT")
    assert captured["symbol"] == "BTC-USDT"
    assert captured["side"] == "sell"
    assert captured["quantity"] == 0.01
    assert captured["order_type"] == "market"


def test_close_position_is_a_noop_when_no_base_balance_held(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        okx_sdk,
        "get_account_snapshot",
        lambda config: _snapshot([{"currency": "USDT", "cash_balance": "100000"}]),
    )

    def _boom(config, **kwargs):
        raise AssertionError("place_order should not be called when there's nothing to flatten")

    monkeypatch.setattr(okx_sdk, "place_order", _boom)
    adapter = vt_okx.OKXExecAdapter(config=_cfg())

    adapter.close_position("BTC-USDT")  # must not raise


def test_close_position_raises_on_snapshot_read_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        okx_sdk, "get_account_snapshot", lambda config: {"status": "error", "error": "rate limited"}
    )
    adapter = vt_okx.OKXExecAdapter(config=_cfg())

    with pytest.raises(vt_okx.OKXAdapterError, match="rate limited"):
        adapter.close_position("BTC-USDT")


def test_close_position_raises_on_flatten_order_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        okx_sdk,
        "get_account_snapshot",
        lambda config: _snapshot([{"currency": "BTC", "cash_balance": "0.01"}]),
    )
    monkeypatch.setattr(
        okx_sdk, "place_order", lambda config, **kwargs: {"status": "error", "error": "size too small"}
    )
    adapter = vt_okx.OKXExecAdapter(config=_cfg())

    with pytest.raises(vt_okx.OKXAdapterError, match="size too small"):
        adapter.close_position("BTC-USDT")


# --------------------------------------------------------------------------- #
# positions -- synthesized from the account balance
# --------------------------------------------------------------------------- #


def test_positions_synthesizes_non_quote_balances_as_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        okx_sdk,
        "get_account_snapshot",
        lambda config: _snapshot(
            [
                {"currency": "USDT", "cash_balance": "95000"},  # quote -> cash, excluded
                {"currency": "BTC", "cash_balance": "0.01"},
                {"currency": "ETH", "cash_balance": "0.5"},
            ]
        ),
    )
    adapter = vt_okx.OKXExecAdapter(config=_cfg())

    rows = adapter.positions()
    by_symbol = {p.symbol: p for p in rows}
    assert "USDT-USDT" not in by_symbol  # the quote is cash, never a position
    assert by_symbol["BTC-USDT"].venue == "okx"
    assert by_symbol["BTC-USDT"].quantity == 0.01
    assert by_symbol["ETH-USDT"].quantity == 0.5


def test_positions_skips_zero_and_dust_balances(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        okx_sdk,
        "get_account_snapshot",
        lambda config: _snapshot(
            [
                {"currency": "BTC", "cash_balance": "0.01"},
                {"currency": "ETH", "cash_balance": "0"},
                {"currency": "DOGE", "cash_balance": "0.0000000000001"},
            ]
        ),
    )
    adapter = vt_okx.OKXExecAdapter(config=_cfg())

    symbols = {p.symbol for p in adapter.positions()}
    assert symbols == {"BTC-USDT"}


def test_positions_is_empty_on_a_cash_only_account(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        okx_sdk,
        "get_account_snapshot",
        lambda config: _snapshot([{"currency": "USDT", "cash_balance": "100000"}]),
    )
    adapter = vt_okx.OKXExecAdapter(config=_cfg())
    assert adapter.positions() == []


def test_positions_raises_on_snapshot_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        okx_sdk, "get_account_snapshot", lambda config: {"status": "error", "error": "timeout"}
    )
    adapter = vt_okx.OKXExecAdapter(config=_cfg())

    with pytest.raises(vt_okx.OKXAdapterError, match="timeout"):
        adapter.positions()


def test_positions_honors_a_non_default_quote_currency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        okx_sdk,
        "get_account_snapshot",
        lambda config: _snapshot(
            [{"currency": "USDC", "cash_balance": "5000"}, {"currency": "SOL", "cash_balance": "10"}]
        ),
    )
    adapter = vt_okx.OKXExecAdapter(config=_cfg(), quote_ccy="USDC")

    rows = adapter.positions()
    assert [p.symbol for p in rows] == ["SOL-USDC"]  # USDC is now the quote, excluded


# --------------------------------------------------------------------------- #
# Integration with vt.exec.adapter -- the adapter satisfies BrokerExecAdapter
# --------------------------------------------------------------------------- #


def test_okx_adapter_satisfies_broker_exec_adapter_protocol_via_submit_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real proof this adapter is usable: run it through
    vt.exec.adapter.submit_atomic exactly as a production caller would,
    not just call its methods directly."""
    from vt.exec import adapter as vt_exec

    monkeypatch.setattr(
        okx_sdk, "place_order", lambda config, **kwargs: {"status": "ok", "order_id": "entry-999"}
    )
    captured: dict[str, Any] = {}
    _install_fake_okx_trade_sdk(monkeypatch, captured=captured)

    adapter = vt_okx.OKXExecAdapter(config=_cfg())
    request = vt_exec.OrderRequest(
        symbol="BTC-USDT",
        side="long",
        size=0.01,
        entry_price=50000.0,
        stop_price=48000.0,
        venue="okx",
        client_order_id="e2ecoid",
    )

    receipt = vt_exec.submit_atomic(adapter, request)

    assert receipt.status == "submitted"
    assert receipt.entry_order_id == "entry-999"
    assert receipt.stop_order_id == "stop-789"
