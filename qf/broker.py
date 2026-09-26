"""QuickFlip's execution surface, and its OKX spot implementation.

Why not reuse `vt.exec.okx.OKXExecAdapter` (Scenario_Spec §1 floated it)?
Three things QuickFlip needs that the VT adapter structurally can't give:

  * **Tagging.** Every QuickFlip order carries a `qf`-prefixed clOrdId
    (Scenario_Spec §2, account isolation). VT's entry leg goes through
    upstream `sdk.place_order`, which drops `clOrdId` entirely.
  * **Exact-quantity exits.** VT's `close_position` sells the account's
    WHOLE base balance. On a shared account (VibeTrading, or the demo
    account's seed balances) that would sell coins QuickFlip never bought.
    QuickFlip sells exactly the size it recorded at entry.
  * **Both exits resident on the exchange.** VT rests a stop only; its
    targets are monitor-driven. QuickFlip holds up to 8 hours, so target
    AND stop go on as one OCO algo order -- a crashed or sleeping monitor
    never leaves the position unmanaged on either side.

So this module talks to python-okx directly (the same lazy-import pattern
`vt/exec/okx.py` already uses for its stop leg), reusing only upstream's
`OKXConfig` for credentials/host/demo-flag. Every call fails closed:
a non-"0" `code`/`sCode` raises `BrokerError` with OKX's own message.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol

if TYPE_CHECKING:  # pragma: no cover
    from src.trading.connectors.okx.sdk import OKXConfig

#: Algo states that mean "still armed on the exchange".
_ALGO_LIVE = frozenset({"live", "pause", "partially_effective"})
_ALGO_TRIGGERED = frozenset({"effective"})
_ALGO_FAILED = frozenset({"order_failed", "partially_failed"})
_ALGO_CANCELLED = frozenset({"canceled"})
#: Order states after which a fill will not grow any further.
_ORDER_DONE = frozenset({"filled", "canceled", "mmp_canceled"})


class BrokerError(RuntimeError):
    """An exchange call failed or returned a rejection."""


@dataclass(frozen=True)
class Instrument:
    lot_sz: float
    min_sz: float
    tick_sz: float


@dataclass(frozen=True)
class Fill:
    state: str
    avg_px: float
    filled_sz: float
    #: Fee charged, as a positive magnitude (OKX reports charges negative).
    fee: float
    fee_ccy: str

    @property
    def done(self) -> bool:
        return self.state in _ORDER_DONE


@dataclass(frozen=True)
class AlgoStatus:
    state: str
    #: The market order the OCO spawned once triggered.
    ord_id: str | None
    #: "tp" or "sl" once triggered.
    side: str | None

    @property
    def live(self) -> bool:
        return self.state in _ALGO_LIVE

    @property
    def triggered(self) -> bool:
        return self.state in _ALGO_TRIGGERED

    @property
    def failed(self) -> bool:
        return self.state in _ALGO_FAILED

    @property
    def cancelled(self) -> bool:
        return self.state in _ALGO_CANCELLED


class Broker(Protocol):
    venue: str
    is_demo: bool
    credentials_source: str

    def taker_rate(self, symbol: str) -> float: ...
    def instrument(self, symbol: str) -> Instrument: ...
    def market_buy_usd(self, symbol: str, usd: float, cl_ord_id: str) -> str: ...
    def market_sell(self, symbol: str, size: float, cl_ord_id: str) -> str: ...
    def cancel_order(self, symbol: str, order_id: str) -> None: ...
    def order_fill(self, symbol: str, order_id: str) -> Fill: ...
    def place_oco(self, symbol: str, size: float, tp_px: float, sl_px: float, algo_cl_ord_id: str) -> str: ...
    def amend_stop(self, symbol: str, algo_id: str, sl_px: float) -> None: ...
    def cancel_algo(self, symbol: str, algo_id: str) -> None: ...
    def algo_status(self, algo_id: str) -> AlgoStatus: ...
    def holding(self, ccy: str) -> float: ...
    def tagged_live_algos(self, prefix: str) -> list[tuple[str, str]]: ...


# --------------------------------------------------------------------------- #
# OKX spot
# --------------------------------------------------------------------------- #


def load_okx_config(dedicated_path: Path) -> tuple["OKXConfig", str]:
    """Dedicated QuickFlip credentials (a demo sub-account) when present,
    else the shared VibeTrading file. Returns (config, source label) so
    `status` can say which isolation mode is actually in effect."""
    from src.trading.connectors.okx import sdk as okx_sdk

    if dedicated_path.exists():
        data = json.loads(dedicated_path.read_text(encoding="utf-8"))
        return okx_sdk.OKXConfig.from_mapping(data), f"dedicated ({dedicated_path})"
    return okx_sdk.load_config(), f"shared ({okx_sdk.config_path()}) -- isolation by qf tag only"


class OKXSpotBroker:
    venue = "okx"

    def __init__(self, config: "OKXConfig", *, credentials_source: str = "", quote_ccy: str = "USDT") -> None:
        self._cfg = config
        self._quote = quote_ccy
        self.credentials_source = credentials_source
        self._trade = None
        self._account = None
        self._public = None
        self._instruments: dict[str, Instrument] = {}

    @property
    def is_demo(self) -> bool:
        return bool(self._cfg.is_demo)

    # -- reads ------------------------------------------------------------ #

    def taker_rate(self, symbol: str) -> float:
        rows = _data(self._account_api().get_fee_rates(instType="SPOT", instId=symbol), "fee_rates")
        # OKX reports the fee you PAY as a negative rate
        return abs(float(rows[0]["taker"]))

    def instrument(self, symbol: str) -> Instrument:
        if symbol not in self._instruments:
            rows = _data(self._public_api().get_instruments(instType="SPOT", instId=symbol), "instruments")
            row = rows[0]
            if row.get("state") != "live":
                raise BrokerError(f"instruments: {symbol} is not live (state={row.get('state')!r})")
            self._instruments[symbol] = Instrument(
                lot_sz=float(row["lotSz"]), min_sz=float(row["minSz"]), tick_sz=float(row["tickSz"])
            )
        return self._instruments[symbol]

    def order_fill(self, symbol: str, order_id: str) -> Fill:
        row = _data(self._trade_api().get_order(instId=symbol, ordId=order_id), "get_order")[0]
        return Fill(
            state=str(row.get("state") or ""),
            avg_px=_num(row.get("avgPx")),
            filled_sz=_num(row.get("accFillSz")),
            fee=-_num(row.get("fee")),
            fee_ccy=str(row.get("feeCcy") or ""),
        )

    def algo_status(self, algo_id: str) -> AlgoStatus:
        row = _data(self._trade_api().get_algo_order_details(algoId=algo_id), "algo_details")[0]
        ord_id = str(row.get("ordId") or "")
        if not ord_id:
            ids = row.get("ordIdList") or []
            ord_id = str(ids[0]) if ids else ""
        return AlgoStatus(
            state=str(row.get("state") or ""),
            ord_id=ord_id or None,
            side=str(row.get("actualSide") or "") or None,
        )

    def holding(self, ccy: str) -> float:
        rows = _data(self._account_api().get_account_balance(ccy=ccy), "balance")
        for detail in rows[0].get("details") or []:
            if str(detail.get("ccy", "")).upper() == ccy.upper():
                return _num(detail.get("cashBal"))
        return 0.0

    def tagged_live_algos(self, prefix: str) -> list[tuple[str, str]]:
        resp = self._trade_api().order_algos_list(ordType="oco", instType="SPOT")
        rows = _data(resp, "algos_list", allow_empty=True)  # no pending OCOs is a normal answer
        return [
            (str(r["instId"]), str(r["algoId"]))
            for r in rows
            if str(r.get("algoClOrdId") or "").startswith(prefix)
        ]

    # -- writes ----------------------------------------------------------- #

    def market_buy_usd(self, symbol: str, usd: float, cl_ord_id: str) -> str:
        resp = self._trade_api().place_order(
            instId=symbol, tdMode="cash", side="buy", ordType="market",
            sz=_fmt(usd), tgtCcy="quote_ccy", clOrdId=cl_ord_id,
        )
        return str(_ok_row(resp, "market_buy")["ordId"])

    def market_sell(self, symbol: str, size: float, cl_ord_id: str) -> str:
        resp = self._trade_api().place_order(
            instId=symbol, tdMode="cash", side="sell", ordType="market",
            sz=_fmt(size), tgtCcy="base_ccy", clOrdId=cl_ord_id,
        )
        return str(_ok_row(resp, "market_sell")["ordId"])

    def cancel_order(self, symbol: str, order_id: str) -> None:
        _ok_row(self._trade_api().cancel_order(instId=symbol, ordId=order_id), "cancel_order")

    def place_oco(self, symbol: str, size: float, tp_px: float, sl_px: float, algo_cl_ord_id: str) -> str:
        resp = self._trade_api().place_algo_order(
            instId=symbol, tdMode="cash", side="sell", ordType="oco", sz=_fmt(size),
            tpTriggerPx=_fmt(tp_px), tpOrdPx="-1", tpTriggerPxType="last",
            slTriggerPx=_fmt(sl_px), slOrdPx="-1", slTriggerPxType="last",
            algoClOrdId=algo_cl_ord_id,
        )
        return str(_ok_row(resp, "place_oco")["algoId"])

    def amend_stop(self, symbol: str, algo_id: str, sl_px: float) -> None:
        resp = self._trade_api().amend_algo_order(
            instId=symbol, algoId=algo_id,
            newSlTriggerPx=_fmt(sl_px), newSlOrdPx="-1", newSlTriggerPxType="last",
        )
        _ok_row(resp, "amend_stop")

    def cancel_algo(self, symbol: str, algo_id: str) -> None:
        _ok_row(self._trade_api().cancel_algo_order([{"instId": symbol, "algoId": algo_id}]), "cancel_algo")

    # -- clients ---------------------------------------------------------- #

    def _trade_api(self):
        if self._trade is None:
            from okx.Trade import TradeAPI  # type: ignore

            c = self._cfg
            self._trade = TradeAPI(c.api_key, c.api_secret, c.passphrase, False, c.flag, domain=c.host)
        return self._trade

    def _account_api(self):
        if self._account is None:
            from okx.Account import AccountAPI  # type: ignore

            c = self._cfg
            self._account = AccountAPI(c.api_key, c.api_secret, c.passphrase, False, c.flag, domain=c.host)
        return self._account

    def _public_api(self):
        if self._public is None:
            from okx.PublicData import PublicAPI  # type: ignore

            self._public = PublicAPI(flag=self._cfg.flag, domain=self._cfg.host)
        return self._public


# --------------------------------------------------------------------------- #
# Response helpers
# --------------------------------------------------------------------------- #


def _data(resp: Any, label: str, *, allow_empty: bool = False) -> list[Mapping[str, Any]]:
    if not isinstance(resp, Mapping) or str(resp.get("code")) != "0":
        msg = resp.get("msg") if isinstance(resp, Mapping) else resp
        raise BrokerError(f"{label}: {msg or 'OKX returned an error'}")
    rows = resp.get("data") or []
    if not rows and not allow_empty:
        raise BrokerError(f"{label}: OKX returned no data")
    return list(rows)


def _ok_row(resp: Any, label: str) -> Mapping[str, Any]:
    """Order-style responses carry a per-row sCode/sMsg that can reject
    even when the envelope code is "0" -- check both."""
    rows = resp.get("data") if isinstance(resp, Mapping) else None
    row = rows[0] if rows else {}
    s_code = str(row.get("sCode", "0") or "0")
    if not rows or str(resp.get("code")) != "0" or s_code != "0":
        msg = row.get("sMsg") or (resp.get("msg") if isinstance(resp, Mapping) else None)
        raise BrokerError(f"{label}: {msg or 'OKX rejected the request'} (sCode={s_code})")
    return row


def _num(value: Any) -> float:
    try:
        return float(value) if value not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: float) -> str:
    """Plain decimal string, no exponent, no float noise."""
    return format(Decimal(repr(float(value))).normalize(), "f")
