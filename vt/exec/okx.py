"""M007 -- Concrete OKX (spot) execution adapter.

Implements `vt.exec.adapter.BrokerExecAdapter` against OKX's demo
(or live) SPOT trading API. Wraps upstream's
`src.trading.connectors.okx.sdk` wherever it can express what M007 needs
-- the same "local import, check `status == 'ok'`, raise on failure"
pattern `vt/data/feed.py` (M001) and `vt/exec/alpaca.py` (M007 Alpaca)
already use (AD001 -- never edit upstream files, import and wrap
instead).

OKX's upstream connector is thinner than Alpaca's, so this adapter has
one more seam than the Alpaca one, and it's the SAME kind of seam:

  * `submit_stop` -- upstream's `place_order` supports only
    `market`/`limit`, no stop-order type. OKX's stop is a *conditional
    algo order* (`TradeAPI.place_algo_order`), which upstream doesn't
    wrap. So the stop leg goes directly against python-okx, lazily
    imported -- exactly the choice the Alpaca adapter made for its own
    stop (a deliberate NON-edit of upstream, not a third AD001
    exception). Everything else -- entry, cancel, flatten, position
    reads -- goes through upstream's tested functions.

Two facts about OKX SPOT shape the rest of this module, and both are
deliberate, documented tradeoffs rather than bugs:

  * **Spot has no "position."** A filled buy simply raises your
    base-currency balance; OKX's `get_positions` returns *derivatives*
    positions, which are always empty for `tdMode="cash"` trading. So
    `positions()` SYNTHESIZES positions from the account balance,
    reporting every non-quote holding as `{ccy}-{quote}` (default quote
    `USDT`). The quote currency itself is cash and is never reported.
    A demo account pre-seeded with test balances will therefore show
    those balances as positions -- which is the honest thing for
    reconciliation to see ("the broker holds something internal state
    doesn't know about" is exactly a drift worth halting on), not
    something to hide here.

  * **Client-order-id idempotency is best-effort.** OKX's `clOrdId`
    charset is alphanumeric, <= 32 chars -- it cannot hold the
    pipeline's derived `<entry_coid>-stop` (a hyphen), and upstream's
    `place_order` wrapper drops `clOrdId` entirely anyway. So the entry
    leg carries no broker-side idempotency key (submit-once safety
    rests on the pipeline calling `submit_atomic` once per approved
    decision, not in a retry loop), and the stop leg forwards the
    caller's id as `algoClOrdId` only when it happens to be OKX-valid,
    otherwise letting OKX assign its own id. The atomicity invariant
    that actually matters -- never a filled entry with no resting stop
    -- is enforced structurally by `submit_atomic`'s compensating
    cancel+close, not by an idempotency key.

SPOT is long-only (`Strategy_Spec.md` sec 3, and you cannot short spot
without margin): `submit_entry`/`submit_stop` reject a `short` side
loudly rather than silently mis-routing it.

Full contract in `03_Modules.md` section M007.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal, Mapping

from vt.exec.adapter import Position

if TYPE_CHECKING:  # pragma: no cover -- type-only, avoids a hard runtime dep
    from src.trading.connectors.okx.sdk import OKXConfig

#: Default quote currency. All Phase-B/C crypto candidates are `-USDT`
#: pairs (`Strategy_Spec.md`, B2/B3); overridable per-adapter for a
#: different quote leg.
DEFAULT_QUOTE_CCY = "USDT"

#: OKX client-order-id charset: alphanumeric, 1-32 chars. Anything else
#: (notably the pipeline's hyphenated `<coid>-stop`) can't be forwarded.
_OKX_CLIENT_ID = re.compile(r"^[A-Za-z0-9]{1,32}$")

#: Balances at or below this magnitude are dust, not a position.
_DUST = 1e-12


class OKXAdapterError(RuntimeError):
    """Raised when an upstream connector call or the raw python-okx algo
    call fails. `submit_atomic`/`reconcile` in `vt.exec.adapter` already
    catch a broad `Exception` at each call site, so this is deliberately
    a plain `RuntimeError` subclass rather than a bespoke hierarchy --
    the caller only needs to know the call failed and see why. Upstream's
    OKX connector fails closed with `{"status": "error", "error": <msg>}`
    rather than raising, so `str(exc)` carries that message verbatim.
    """


class OKXExecAdapter:
    """Concrete `BrokerExecAdapter` for OKX spot. Constructed with an
    optional upstream `OKXConfig`; when omitted, loads
    `~/.vibe-trading/okx.json` via upstream's own `load_config()` -- the
    same credentials file every other OKX-touching piece of this project
    already uses (S007).

    `quote_ccy` is the currency treated as cash (never reported as a
    position, and the leg every traded pair is denominated in). Defaults
    to `USDT`.

    Not a dataclass, for the same reason the Alpaca adapter isn't: the
    "load from disk only when config is None" branch is construction-time
    logic a dataclass default factory can't express without eagerly
    loading config even when the caller supplied one.
    """

    venue = "okx"

    def __init__(self, config: "OKXConfig | None" = None, *, quote_ccy: str = DEFAULT_QUOTE_CCY) -> None:
        if config is None:
            from src.trading.connectors.okx import sdk as okx_sdk

            config = okx_sdk.load_config()
        self._config = config
        self._quote_ccy = quote_ccy.strip().upper()

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
        """Limit BUY at the entry price (`Strategy_Spec.md` sec 4: limit
        at the retest, not a market chase). Spot is long-only, so `side`
        is always `long` here -- a `short` is rejected rather than
        silently mis-routed. `client_order_id` is accepted for Protocol
        symmetry but not forwarded (see module docstring: upstream's
        `place_order` drops it, and OKX's charset can't hold the
        pipeline's ids anyway).
        """
        if side != "long":
            raise OKXAdapterError(
                f"submit_entry: OKX spot is long-only (Strategy_Spec sec 3); got side={side!r}"
            )
        from src.trading.connectors.okx import sdk as okx_sdk

        result = okx_sdk.place_order(
            self._config,
            symbol=symbol,
            side="buy",
            quantity=size,
            order_type="limit",
            limit_price=limit_price,
        )
        if result.get("status") != "ok":
            raise OKXAdapterError(f"submit_entry: {result.get('error')}")
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
        """Hard stop, submitted directly via python-okx's
        `TradeAPI.place_algo_order` (upstream has no stop-order type --
        see module docstring). A long spot position is stopped by a
        CONDITIONAL SELL that fires a market order (`slOrdPx="-1"`) when
        the last price reaches `stop_price`. Spot is long-only, so a
        `short` side is rejected.

        The caller's `client_order_id` is forwarded as `algoClOrdId`
        only when it is OKX-valid (alphanumeric, <= 32); otherwise it is
        omitted and OKX assigns its own id (the returned `algoId` is what
        the caller gets back either way).
        """
        if side != "long":
            raise OKXAdapterError(
                f"submit_stop: OKX spot is long-only (Strategy_Spec sec 3); got side={side!r}"
            )

        params: dict[str, Any] = {
            "instId": symbol,
            "tdMode": "cash",
            "side": "sell",  # closing side of a long spot position
            "ordType": "conditional",
            "sz": str(size),
            "slTriggerPx": str(stop_price),
            "slOrdPx": "-1",  # OKX convention: execute as a market order on trigger
            "slTriggerPxType": "last",
        }
        algo_client_id = client_order_id if _OKX_CLIENT_ID.match(client_order_id or "") else ""
        if algo_client_id:
            params["algoClOrdId"] = algo_client_id

        client = self._trade_client()
        try:
            resp = client.place_algo_order(**params)
        except Exception as exc:  # noqa: BLE001 -- surface every broker failure uniformly
            raise OKXAdapterError(f"submit_stop: {exc!r}") from exc
        return self._require_ok_order(resp, id_field="algoId", label="submit_stop")

    def cancel(self, order_id: str) -> None:
        """Cancel a resting order. OKX's cancel requires the instrument
        id, which the Protocol's `cancel(order_id)` doesn't carry -- so
        look the symbol up from the open-orders list first. An order that
        isn't in the open list (already filled, already cancelled, never
        placed) raises: `submit_atomic`'s compensating path treats a
        cancel failure as non-fatal and also runs `close_position`, so a
        filled entry still gets flattened.
        """
        from src.trading.connectors.okx import sdk as okx_sdk

        listing = okx_sdk.get_open_orders(self._config)
        if listing.get("status") != "ok":
            raise OKXAdapterError(f"cancel: {listing.get('error')}")

        row = next(
            (o for o in listing.get("open_orders", []) if str(o.get("order_id")) == str(order_id)),
            None,
        )
        if row is None:
            raise OKXAdapterError(f"cancel: order {order_id!r} not found among open orders")

        result = okx_sdk.cancel_order(self._config, order_id=str(order_id), symbol=row.get("symbol"))
        if result.get("status") != "ok":
            raise OKXAdapterError(f"cancel: {result.get('error')}")

    def close_position(self, symbol: str) -> None:
        """Flatten a long spot position by market-selling the held
        base-currency balance. A no-op when nothing is held -- the
        compensating close in `submit_atomic` may run even when the entry
        never actually filled, and closing nothing is the correct
        response to that, not an error.
        """
        from src.trading.connectors.okx import sdk as okx_sdk

        base = self._base_ccy(symbol)
        held = self._held_balance(base)
        if held <= _DUST:
            return

        result = okx_sdk.place_order(
            self._config,
            symbol=symbol,
            side="sell",
            quantity=held,
            order_type="market",
        )
        if result.get("status") != "ok":
            raise OKXAdapterError(f"close_position: {result.get('error')}")

    def positions(self) -> list[Position]:
        """Broker's view of what's held, synthesized from the account
        balance (spot has no native position -- see module docstring).
        One row per non-quote currency with a non-dust holding, keyed by
        `{ccy}-{quote}` so it lines up with how internal state keys a
        position it opened.
        """
        details = self._balance_details()
        rows: list[Position] = []
        for detail in details:
            ccy = str(detail.get("currency") or "").strip().upper()
            if not ccy or ccy == self._quote_ccy:
                continue
            qty = _coerce_balance(detail)
            if qty <= _DUST:
                continue
            rows.append(Position(venue=self.venue, symbol=f"{ccy}-{self._quote_ccy}", quantity=qty))
        return rows

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _base_ccy(self, symbol: str) -> str:
        """`BTC-USDT` -> `BTC`. Falls back to the whole symbol (uppercased)
        when it isn't hyphen-delimited, so a malformed symbol fails at the
        balance lookup rather than here."""
        return str(symbol or "").strip().upper().split("-")[0]

    def _balance_details(self) -> list[dict[str, Any]]:
        from src.trading.connectors.okx import sdk as okx_sdk

        snapshot = okx_sdk.get_account_snapshot(self._config)
        if snapshot.get("status") != "ok":
            raise OKXAdapterError(f"positions: {snapshot.get('error')}")
        account = snapshot.get("account") or {}
        details = account.get("details") or []
        return list(details)

    def _held_balance(self, base: str) -> float:
        for detail in self._balance_details():
            if str(detail.get("currency") or "").strip().upper() == base:
                return _coerce_balance(detail)
        return 0.0

    def _require_ok_order(self, resp: Any, *, id_field: str, label: str) -> str:
        """Interpret an OKX order/algo response, failing closed on any
        non-zero code. Success requires both the outer `code == "0"` and
        the first data row's `sCode == "0"`; otherwise the row's `sMsg`
        (or the top-level `msg`) is surfaced. Mirrors upstream's private
        `_order_result` for the fields this direct-call path needs.
        """
        rows = resp.get("data") if isinstance(resp, Mapping) else None
        if not rows:
            msg = resp.get("msg") if isinstance(resp, Mapping) else None
            raise OKXAdapterError(f"{label}: {msg or 'OKX returned no order data'}")
        row = rows[0]
        s_code = str(row.get("sCode") or "")
        if s_code != "0":
            msg = row.get("sMsg") or (resp.get("msg") if isinstance(resp, Mapping) else None)
            raise OKXAdapterError(f"{label}: {msg or f'OKX rejected order (sCode={s_code or chr(63)})'}")
        return str(row.get(id_field) or "")

    def _trade_client(self):
        """Construct a python-okx `TradeAPI`, matching how upstream's
        connector constructs its own clients (`api_key, api_secret,
        passphrase, use_server_time=False, flag, domain=host`).
        Deliberately not importing upstream's private `_trade_client` --
        the leading underscore signals "not a stable API to depend on."
        """
        from okx.Trade import TradeAPI  # type: ignore

        return TradeAPI(
            self._config.api_key,
            self._config.api_secret,
            self._config.passphrase,
            False,
            self._config.flag,
            domain=self._config.host,
        )


def _coerce_balance(detail: Mapping[str, Any]) -> float:
    """Read a holding out of a balance detail row, preferring cash balance
    (total held), then equity, then available. Any non-numeric / missing
    value reads as 0.0 rather than raising -- a garbled row means "treat
    as no holding," which is the safe direction for both flatten and
    reconcile.
    """
    for key in ("cash_balance", "equity", "available"):
        value = detail.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0
