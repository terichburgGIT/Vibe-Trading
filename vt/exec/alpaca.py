"""M007 -- Concrete Alpaca execution adapter.

Implements `vt.exec.adapter.BrokerExecAdapter` against Alpaca's paper
(or live) trading API. Wraps upstream's `src.trading.connectors.alpaca.sdk`
for entry submission, cancel, and position reads -- the same pattern
`vt/data/feed.py` (M001) already uses for market data: local import
inside each method, check `result["status"] == "ok"`, raise on failure
(AD001 -- never edit upstream files, import and wrap instead).

The one thing upstream's `sdk.place_order` doesn't support is a STOP
order -- Strategy_Spec.md's hard stop (`entry - 1.5*ATR`, submitted
with entry, never widened) needs a resting stop order, and upstream's
`order_type` is restricted to `"market"`/`"limit"` only. Rather than
editing upstream's sdk.py to add a third order type (a third AD001
exception, after the two already logged for M001/M002 -- see
`07_Architecture_Decisions.md`), this module submits stop orders
directly against alpaca-py's SDK, lazily imported -- the same pattern
`vt/alerts/kill.py` already uses for its own broker calls. Everything
else (entry, cancel, position reads, and the market order used to
flatten a position) goes through upstream's tested functions
unchanged.

Full contract in `03_Modules.md` section M007.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from vt.exec.adapter import Position

if TYPE_CHECKING:  # pragma: no cover -- type-only, avoids a hard runtime dep
    from src.trading.connectors.alpaca.sdk import AlpacaConfig


class AlpacaAdapterError(RuntimeError):
    """Raised when an upstream connector call or the raw alpaca-py SDK
    call fails. `submit_atomic`/`reconcile` in `vt.exec.adapter` already
    catch a broad `Exception` at each call site, so this is deliberately
    a plain `RuntimeError` subclass rather than a bespoke hierarchy --
    the caller doesn't need to distinguish failure modes, only to know
    the call failed and see why (`str(exc)` carries the upstream error
    message verbatim, since upstream's connector functions fail closed
    with `{"status": "error", "error": <message>}` rather than raising).
    """


class AlpacaExecAdapter:
    """Concrete `BrokerExecAdapter` for Alpaca. Constructed with an
    optional upstream `AlpacaConfig`; when omitted, loads
    `~/.vibe-trading/alpaca.json` via upstream's own `load_config()` --
    the same credentials file every other Alpaca-touching piece of this
    project already uses (S007).

    Not a dataclass on purpose: a dataclass-generated `__init__` would
    expose the config param under its private-looking storage name,
    and this class needs one line of construction-time logic (the
    "load from disk if not supplied" branch) that a dataclass default
    factory can't express without eagerly loading config even when the
    caller means to supply one.
    """

    venue = "alpaca"

    def __init__(self, config: "AlpacaConfig | None" = None) -> None:
        if config is None:
            from src.trading.connectors.alpaca import sdk as alpaca_sdk

            config = alpaca_sdk.load_config()
        self._config = config

    # ------------------------------------------------------------------ #
    # BrokerExecAdapter Protocol
    # ------------------------------------------------------------------ #

    def submit_entry(
        self,
        *,
        symbol: str,
        side: Literal["long", "short"],
        size: float,
        limit_price: float,
        client_order_id: str,
    ) -> str:
        """Limit order at the entry price (`Strategy_Spec.md` sec 4:
        "Limit order at the retest of the breakout level, not a market
        chase"). `side` is the POSITION's side -- long opens with a
        buy, short opens with a sell.
        """
        from src.trading.connectors.alpaca import sdk as alpaca_sdk

        result = alpaca_sdk.place_order(
            self._config,
            symbol=symbol,
            side="buy" if side == "long" else "sell",
            quantity=size,
            order_type="limit",
            limit_price=limit_price,
            client_order_id=client_order_id,
        )
        if result.get("status") != "ok":
            raise AlpacaAdapterError(f"submit_entry: {result.get('error')}")
        return str(result["order_id"])

    def submit_stop(
        self,
        *,
        symbol: str,
        side: Literal["long", "short"],
        size: float,
        stop_price: float,
        client_order_id: str,
    ) -> str:
        """Hard stop, submitted directly via alpaca-py (upstream's
        `sdk.place_order` has no stop-order type -- see module
        docstring). `side` is the POSITION's side, same convention as
        `submit_entry` -- the actual order side is the CLOSING side: a
        long position's stop is a sell, a short position's stop is a
        buy. Time-in-force is GTC (a day-only stop would expire and
        leave the position unstopped overnight, which is exactly the
        state `Risk_Policy.md` treats as worst-case).

        Note: `vt.exec.adapter.submit_atomic` derives the stop's
        `client_order_id` as `<entry_coid>-stop`. Alpaca caps
        client_order_id at 48 characters (enforced by upstream's own
        `place_order` validation, mirrored here since this call bypasses
        that function) -- a caller supplying an entry coid near that
        limit will get a clean, catchable failure here rather than a
        silent truncation, and `submit_atomic` already treats any stop
        failure as "flatten the entry," which is the correct response.
        """
        if not (1 <= len(client_order_id) <= 48):
            raise AlpacaAdapterError(
                f"submit_stop: client_order_id must contain 1-48 characters, got {len(client_order_id)}"
            )

        client = self._trading_client()
        from alpaca.trading.enums import OrderSide, TimeInForce  # type: ignore
        from alpaca.trading.requests import StopOrderRequest  # type: ignore

        stop_side = OrderSide.SELL if side == "long" else OrderSide.BUY
        req = StopOrderRequest(
            symbol=symbol,
            qty=size,
            side=stop_side,
            time_in_force=TimeInForce.GTC,
            stop_price=stop_price,
            client_order_id=client_order_id,
        )
        try:
            order = client.submit_order(order_data=req)
        except Exception as exc:  # noqa: BLE001 -- surface every broker failure uniformly
            raise AlpacaAdapterError(f"submit_stop: {exc!r}") from exc
        return str(getattr(order, "id", ""))

    def cancel(self, order_id: str) -> None:
        from src.trading.connectors.alpaca import sdk as alpaca_sdk

        result = alpaca_sdk.cancel_order(self._config, order_id=order_id)
        if result.get("status") != "ok":
            raise AlpacaAdapterError(f"cancel: {result.get('error')}")

    def close_position(self, symbol: str) -> None:
        """Alpaca's dedicated close-position endpoint isn't wrapped by
        upstream either -- flatten via a market order in the opposite
        direction for the current position size instead, fully
        expressible through upstream's existing `place_order` (no new
        capability, no AD001 exception needed). A no-op when there is
        no open position for `symbol` -- `submit_atomic`'s compensating
        close is defensive and may run even when the entry never
        actually filled, and closing nothing is the correct response to
        that, not an error.
        """
        from src.trading.connectors.alpaca import sdk as alpaca_sdk

        result = alpaca_sdk.get_positions(self._config)
        if result.get("status") != "ok":
            raise AlpacaAdapterError(f"close_position: {result.get('error')}")

        row = next((p for p in result["positions"] if p.get("symbol") == symbol), None)
        if row is None:
            return
        qty = float(row.get("quantity") or 0.0)
        if qty == 0.0:
            return

        flatten_side = "sell" if qty > 0 else "buy"
        close_result = alpaca_sdk.place_order(
            self._config,
            symbol=symbol,
            side=flatten_side,
            quantity=abs(qty),
            order_type="market",
        )
        if close_result.get("status") != "ok":
            raise AlpacaAdapterError(f"close_position: {close_result.get('error')}")

    def positions(self) -> list[Position]:
        from src.trading.connectors.alpaca import sdk as alpaca_sdk

        result = alpaca_sdk.get_positions(self._config)
        if result.get("status") != "ok":
            raise AlpacaAdapterError(f"positions: {result.get('error')}")
        return [
            Position(venue=self.venue, symbol=row["symbol"], quantity=float(row.get("quantity") or 0.0))
            for row in result["positions"]
        ]

    # ------------------------------------------------------------------ #
    # SDK plumbing
    # ------------------------------------------------------------------ #

    def _trading_client(self):
        """Own small `TradingClient` constructor -- deliberately not
        importing upstream's private `_trading_client` (the leading
        underscore signals "not a stable API to depend on" even though
        nothing stops a same-language import). Three lines, matches
        upstream's own construction exactly (`api_key, secret_key,
        paper=cfg.is_paper`).
        """
        from alpaca.trading.client import TradingClient  # type: ignore

        return TradingClient(self._config.api_key, self._config.secret_key, paper=self._config.is_paper)
