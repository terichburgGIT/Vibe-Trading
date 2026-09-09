"""Tests for the concrete Alpaca execution adapter (vt.exec.alpaca),
which fills in M007's `BrokerExecAdapter` Protocol against Alpaca's
paper/live trading API.

Two mocking strategies, matching the two upstream surfaces this module
touches:
  * Upstream's `src.trading.connectors.alpaca.sdk` functions
    (`place_order`, `cancel_order`, `get_positions`, `load_config`) are
    monkeypatched directly on the module object -- same pattern
    `test_feed.py` uses for M001.
  * Raw alpaca-py (`alpaca.trading.client.TradingClient`,
    `.requests.StopOrderRequest`, `.enums.OrderSide`/`TimeInForce`) is
    faked via `sys.modules` injection -- same pattern `test_kill.py`
    uses for its lazy-imported broker SDK calls.

Neither strategy touches a real network call or a real Alpaca account.

Run with: pytest vt/tests -m unit
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from src.trading.connectors.alpaca import sdk as alpaca_sdk
from vt.exec import alpaca as vt_alpaca

pytestmark = pytest.mark.unit


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        api_key="PKTESTKEY",
        secret_key="sk-test-XXXX",
        profile="paper",
        feed="iex",
        is_paper=True,
    )


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_venue_is_alpaca() -> None:
    adapter = vt_alpaca.AlpacaExecAdapter(config=_cfg())
    assert adapter.venue == "alpaca"


def test_constructing_without_config_loads_from_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alpaca_sdk, "load_config", lambda: _cfg())
    adapter = vt_alpaca.AlpacaExecAdapter()
    assert adapter._config.api_key == "PKTESTKEY"


def test_constructing_with_explicit_config_does_not_touch_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom():
        raise AssertionError("load_config should not be called when config is supplied")

    monkeypatch.setattr(alpaca_sdk, "load_config", _boom)
    adapter = vt_alpaca.AlpacaExecAdapter(config=_cfg())
    assert adapter._config.api_key == "PKTESTKEY"


# --------------------------------------------------------------------------- #
# submit_entry
# --------------------------------------------------------------------------- #


def test_submit_entry_long_places_a_buy_limit_order(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_place_order(config, **kwargs):
        captured.update(kwargs)
        captured["config"] = config
        return {"status": "ok", "order_id": "entry-123"}

    monkeypatch.setattr(alpaca_sdk, "place_order", fake_place_order)
    adapter = vt_alpaca.AlpacaExecAdapter(config=_cfg())

    order_id = adapter.submit_entry(
        symbol="AAPL", side="long", size=13.0, limit_price=175.0, client_order_id="coid-1"
    )

    assert order_id == "entry-123"
    assert captured["symbol"] == "AAPL"
    assert captured["side"] == "buy"
    assert captured["quantity"] == 13.0
    assert captured["order_type"] == "limit"
    assert captured["limit_price"] == 175.0
    assert captured["client_order_id"] == "coid-1"


def test_submit_entry_short_places_a_sell_limit_order(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_place_order(config, **kwargs):
        captured.update(kwargs)
        return {"status": "ok", "order_id": "entry-456"}

    monkeypatch.setattr(alpaca_sdk, "place_order", fake_place_order)
    adapter = vt_alpaca.AlpacaExecAdapter(config=_cfg())

    adapter.submit_entry(symbol="TSLA", side="short", size=5.0, limit_price=200.0, client_order_id="coid-2")
    assert captured["side"] == "sell"


def test_submit_entry_raises_alpaca_adapter_error_on_broker_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        alpaca_sdk, "place_order", lambda config, **kwargs: {"status": "error", "error": "insufficient buying power"}
    )
    adapter = vt_alpaca.AlpacaExecAdapter(config=_cfg())

    with pytest.raises(vt_alpaca.AlpacaAdapterError, match="insufficient buying power"):
        adapter.submit_entry(symbol="AAPL", side="long", size=13.0, limit_price=175.0, client_order_id="coid-3")


# --------------------------------------------------------------------------- #
# submit_stop -- raw alpaca-py, faked via sys.modules
# --------------------------------------------------------------------------- #


def _install_fake_alpaca_trading_sdk(monkeypatch: pytest.MonkeyPatch, *, captured: dict[str, Any]) -> None:
    """Install a fake `alpaca.trading.{client,requests,enums}` tree so
    `submit_stop`'s lazy imports resolve to test doubles instead of the
    real alpaca-py package."""

    class FakeOrder:
        def __init__(self, order_id: str) -> None:
            self.id = order_id

    class FakeTradingClient:
        def __init__(self, api_key: str, secret_key: str, paper: bool = True) -> None:
            captured["api_key"] = api_key
            captured["secret_key"] = secret_key
            captured["paper"] = paper

        def submit_order(self, *, order_data):
            captured["order_data"] = order_data
            if captured.get("fail_submit"):
                raise RuntimeError("stop submit refused by broker")
            return FakeOrder("stop-789")

    class FakeStopOrderRequest:
        def __init__(self, **kwargs) -> None:
            captured["stop_request_kwargs"] = kwargs
            self.__dict__.update(kwargs)

    fake_alpaca = types.ModuleType("alpaca")
    fake_trading = types.ModuleType("alpaca.trading")
    fake_client_mod = types.ModuleType("alpaca.trading.client")
    fake_client_mod.TradingClient = FakeTradingClient
    fake_requests_mod = types.ModuleType("alpaca.trading.requests")
    fake_requests_mod.StopOrderRequest = FakeStopOrderRequest
    fake_enums_mod = types.ModuleType("alpaca.trading.enums")
    fake_enums_mod.OrderSide = SimpleNamespace(BUY="buy", SELL="sell")
    fake_enums_mod.TimeInForce = SimpleNamespace(GTC="gtc", DAY="day")

    fake_alpaca.trading = fake_trading
    fake_trading.client = fake_client_mod
    fake_trading.requests = fake_requests_mod
    fake_trading.enums = fake_enums_mod

    monkeypatch.setitem(sys.modules, "alpaca", fake_alpaca)
    monkeypatch.setitem(sys.modules, "alpaca.trading", fake_trading)
    monkeypatch.setitem(sys.modules, "alpaca.trading.client", fake_client_mod)
    monkeypatch.setitem(sys.modules, "alpaca.trading.requests", fake_requests_mod)
    monkeypatch.setitem(sys.modules, "alpaca.trading.enums", fake_enums_mod)


def test_submit_stop_for_a_long_position_submits_a_sell(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _install_fake_alpaca_trading_sdk(monkeypatch, captured=captured)
    adapter = vt_alpaca.AlpacaExecAdapter(config=_cfg())

    order_id = adapter.submit_stop(
        symbol="AAPL", side="long", size=13.0, stop_price=172.5, client_order_id="coid-1-stop"
    )

    assert order_id == "stop-789"
    assert captured["api_key"] == "PKTESTKEY"
    assert captured["paper"] is True
    kwargs = captured["stop_request_kwargs"]
    assert kwargs["side"] == "sell"  # long position -> closing side is sell
    assert kwargs["symbol"] == "AAPL"
    assert kwargs["qty"] == 13.0
    assert kwargs["stop_price"] == 172.5
    assert kwargs["time_in_force"] == "gtc"
    assert kwargs["client_order_id"] == "coid-1-stop"


def test_submit_stop_for_a_short_position_submits_a_buy(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _install_fake_alpaca_trading_sdk(monkeypatch, captured=captured)
    adapter = vt_alpaca.AlpacaExecAdapter(config=_cfg())

    adapter.submit_stop(symbol="TSLA", side="short", size=5.0, stop_price=210.0, client_order_id="coid-2-stop")
    assert captured["stop_request_kwargs"]["side"] == "buy"  # short position -> closing side is buy


def test_submit_stop_raises_alpaca_adapter_error_on_broker_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {"fail_submit": True}
    _install_fake_alpaca_trading_sdk(monkeypatch, captured=captured)
    adapter = vt_alpaca.AlpacaExecAdapter(config=_cfg())

    with pytest.raises(vt_alpaca.AlpacaAdapterError, match="stop submit refused"):
        adapter.submit_stop(symbol="AAPL", side="long", size=13.0, stop_price=172.5, client_order_id="coid-3-stop")


def test_submit_stop_rejects_a_client_order_id_over_48_chars(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _install_fake_alpaca_trading_sdk(monkeypatch, captured=captured)
    adapter = vt_alpaca.AlpacaExecAdapter(config=_cfg())

    too_long = "x" * 49
    with pytest.raises(vt_alpaca.AlpacaAdapterError, match="1-48 characters"):
        adapter.submit_stop(symbol="AAPL", side="long", size=13.0, stop_price=172.5, client_order_id=too_long)
    # And it never even reached the broker call.
    assert "order_data" not in captured


# --------------------------------------------------------------------------- #
# cancel
# --------------------------------------------------------------------------- #


def test_cancel_delegates_to_upstream_cancel_order(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_cancel_order(config, *, order_id):
        captured["order_id"] = order_id
        return {"status": "ok"}

    monkeypatch.setattr(alpaca_sdk, "cancel_order", fake_cancel_order)
    adapter = vt_alpaca.AlpacaExecAdapter(config=_cfg())

    adapter.cancel("entry-123")
    assert captured["order_id"] == "entry-123"


def test_cancel_raises_on_broker_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        alpaca_sdk, "cancel_order", lambda config, *, order_id: {"status": "error", "error": "order not found"}
    )
    adapter = vt_alpaca.AlpacaExecAdapter(config=_cfg())

    with pytest.raises(vt_alpaca.AlpacaAdapterError, match="order not found"):
        adapter.cancel("nonexistent")


# --------------------------------------------------------------------------- #
# close_position
# --------------------------------------------------------------------------- #


def test_close_position_flattens_a_long_position_with_a_market_sell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        alpaca_sdk,
        "get_positions",
        lambda config: {"status": "ok", "positions": [{"symbol": "AAPL", "quantity": 13.0}]},
    )
    captured: dict[str, Any] = {}

    def fake_place_order(config, **kwargs):
        captured.update(kwargs)
        return {"status": "ok", "order_id": "flatten-1"}

    monkeypatch.setattr(alpaca_sdk, "place_order", fake_place_order)
    adapter = vt_alpaca.AlpacaExecAdapter(config=_cfg())

    adapter.close_position("AAPL")
    assert captured["side"] == "sell"
    assert captured["quantity"] == 13.0
    assert captured["order_type"] == "market"


def test_close_position_flattens_a_short_position_with_a_market_buy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        alpaca_sdk,
        "get_positions",
        lambda config: {"status": "ok", "positions": [{"symbol": "TSLA", "quantity": -5.0}]},
    )
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        alpaca_sdk,
        "place_order",
        lambda config, **kwargs: (captured.update(kwargs), {"status": "ok", "order_id": "flatten-2"})[1],
    )
    adapter = vt_alpaca.AlpacaExecAdapter(config=_cfg())

    adapter.close_position("TSLA")
    assert captured["side"] == "buy"
    assert captured["quantity"] == 5.0  # magnitude, not signed


def test_close_position_is_a_noop_when_no_position_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alpaca_sdk, "get_positions", lambda config: {"status": "ok", "positions": []})

    def _boom(config, **kwargs):
        raise AssertionError("place_order should not be called when there's nothing to flatten")

    monkeypatch.setattr(alpaca_sdk, "place_order", _boom)
    adapter = vt_alpaca.AlpacaExecAdapter(config=_cfg())

    adapter.close_position("AAPL")  # must not raise


def test_close_position_raises_on_positions_read_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alpaca_sdk, "get_positions", lambda config: {"status": "error", "error": "auth failed"})
    adapter = vt_alpaca.AlpacaExecAdapter(config=_cfg())

    with pytest.raises(vt_alpaca.AlpacaAdapterError, match="auth failed"):
        adapter.close_position("AAPL")


def test_close_position_raises_on_flatten_order_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        alpaca_sdk,
        "get_positions",
        lambda config: {"status": "ok", "positions": [{"symbol": "AAPL", "quantity": 13.0}]},
    )
    monkeypatch.setattr(
        alpaca_sdk, "place_order", lambda config, **kwargs: {"status": "error", "error": "market closed"}
    )
    adapter = vt_alpaca.AlpacaExecAdapter(config=_cfg())

    with pytest.raises(vt_alpaca.AlpacaAdapterError, match="market closed"):
        adapter.close_position("AAPL")


# --------------------------------------------------------------------------- #
# positions
# --------------------------------------------------------------------------- #


def test_positions_maps_upstream_rows_to_vt_position_dataclass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        alpaca_sdk,
        "get_positions",
        lambda config: {
            "status": "ok",
            "positions": [
                {"symbol": "AAPL", "quantity": 13.0},
                {"symbol": "TSLA", "quantity": -5.0},
            ],
        },
    )
    adapter = vt_alpaca.AlpacaExecAdapter(config=_cfg())

    rows = adapter.positions()
    assert len(rows) == 2
    assert rows[0].venue == "alpaca"
    assert rows[0].symbol == "AAPL"
    assert rows[0].quantity == 13.0
    assert rows[1].symbol == "TSLA"
    assert rows[1].quantity == -5.0


def test_positions_raises_on_broker_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alpaca_sdk, "get_positions", lambda config: {"status": "error", "error": "timeout"})
    adapter = vt_alpaca.AlpacaExecAdapter(config=_cfg())

    with pytest.raises(vt_alpaca.AlpacaAdapterError, match="timeout"):
        adapter.positions()


def test_positions_treats_missing_quantity_as_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        alpaca_sdk,
        "get_positions",
        lambda config: {"status": "ok", "positions": [{"symbol": "AAPL", "quantity": None}]},
    )
    adapter = vt_alpaca.AlpacaExecAdapter(config=_cfg())

    rows = adapter.positions()
    assert rows[0].quantity == 0.0


# --------------------------------------------------------------------------- #
# Integration with vt.exec.adapter -- the adapter satisfies BrokerExecAdapter
# --------------------------------------------------------------------------- #


def test_alpaca_adapter_satisfies_broker_exec_adapter_protocol_via_submit_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real proof this adapter is usable: run it through
    vt.exec.adapter.submit_atomic exactly as a production caller would,
    not just call its methods directly."""
    from vt.exec import adapter as vt_exec

    monkeypatch.setattr(
        alpaca_sdk, "place_order", lambda config, **kwargs: {"status": "ok", "order_id": "entry-999"}
    )
    captured: dict[str, Any] = {}
    _install_fake_alpaca_trading_sdk(monkeypatch, captured=captured)

    adapter = vt_alpaca.AlpacaExecAdapter(config=_cfg())
    request = vt_exec.OrderRequest(
        symbol="AAPL",
        side="long",
        size=13.0,
        entry_price=175.0,
        stop_price=172.5,
        venue="alpaca",
        client_order_id="e2e-coid",
    )

    receipt = vt_exec.submit_atomic(adapter, request)

    assert receipt.status == "submitted"
    assert receipt.entry_order_id == "entry-999"
    assert receipt.stop_order_id == "stop-789"
