"""Shared fixtures for qf tests: an in-memory fake OKX broker, a settable
clock, and synthetic hourly bars (5-point range on a 100 close ->
ATR1h 5%, so k=2 gives a 10% target / 5% stop, above the fee floor)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from qf.broker import AlgoStatus, BrokerError, Fill, Instrument
from qf.config import QFConfig
from qf.engine import QuickFlip
from vt.data.feed import Bar, Quote

T0 = datetime(2026, 9, 25, 12, 0, 30, tzinfo=timezone.utc)
SYM = "SOL-USDT"


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class FakeBroker:
    venue = "okx"
    credentials_source = "fake"

    def __init__(self) -> None:
        self.is_demo = True
        self.calls: list[tuple] = []
        self.fail: set[str] = set()
        self.px = 100.0
        self.holdings = {"SOL": 1_000_000.0}  # far more than QuickFlip bought
        self.orders: dict[str, Fill] = {}
        self.algos: dict[str, AlgoStatus] = {}
        self.algo_symbols: dict[str, str] = {}
        self.algo_client_ids: dict[str, str] = {}
        #: order states the next market sell / buy reports (default filled)
        self.sell_state = "filled"
        self.buy_states: list[str] = []
        self.on_buy = None
        self._n = 0

    def _next(self, kind: str) -> str:
        self._n += 1
        return f"{kind}{self._n}"

    def _call(self, name: str, *args) -> None:
        self.calls.append((name, *args))
        if name in self.fail:
            raise BrokerError(f"{name} boom")

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]

    def taker_rate(self, symbol):
        return 0.0035

    def instrument(self, symbol):
        return Instrument(lot_sz=0.000001, min_sz=0.001, tick_sz=0.01)

    def market_buy_usd(self, symbol, usd, cl_ord_id):
        if self.on_buy:
            self.on_buy()
        self._call("market_buy_usd", symbol, usd, cl_ord_id)
        oid = self._next("o")
        qty = usd / self.px
        state = self.buy_states.pop(0) if self.buy_states else "filled"
        self.orders[oid] = Fill(state, self.px, qty, qty * 0.0035, "SOL")
        return oid

    def cancel_order(self, symbol, order_id):
        self._call("cancel_order", symbol, order_id)
        fill = self.orders[order_id]
        if fill.state == "partially_filled":
            self.orders[order_id] = Fill("canceled", fill.avg_px, fill.filled_sz, fill.fee, fill.fee_ccy)

    def market_sell(self, symbol, size, cl_ord_id):
        self._call("market_sell", symbol, size, cl_ord_id)
        oid = self._next("o")
        filled = size if self.sell_state == "filled" else 0.0
        self.orders[oid] = Fill(self.sell_state, self.px, filled, filled * self.px * 0.0035, "USDT")
        return oid

    def order_fill(self, symbol, order_id):
        return self.orders[order_id]

    def place_oco(self, symbol, size, tp_px, sl_px, algo_cl_ord_id):
        self._call("place_oco", symbol, size, tp_px, sl_px, algo_cl_ord_id)
        aid = self._next("a")
        self.algos[aid] = AlgoStatus("live", None, None)
        self.algo_symbols[aid] = symbol
        self.algo_client_ids[aid] = algo_cl_ord_id
        return aid

    def amend_stop(self, symbol, algo_id, sl_px):
        self._call("amend_stop", symbol, algo_id, sl_px)

    def cancel_algo(self, symbol, algo_id):
        self._call("cancel_algo", symbol, algo_id)
        self.algos[algo_id] = AlgoStatus("canceled", None, None)

    def algo_status(self, algo_id):
        return self.algos[algo_id]

    def holding(self, ccy):
        self._call("holding", ccy)
        return self.holdings.get(ccy, 0.0)

    def tagged_live_algos(self, prefix):
        self._call("tagged_live_algos", prefix)
        return [
            (self.algo_symbols[a], a)
            for a, s in self.algos.items()
            if s.live and self.algo_client_ids.get(a, "").startswith(prefix)
        ]

    # test helper: the exchange fires one side of an OCO
    def trigger(self, algo_id, side, px, size):
        oid = self._next("o")
        self.orders[oid] = Fill("filled", px, size, size * px * 0.0035, "USDT")
        self.algos[algo_id] = AlgoStatus("effective", oid, side)


def _bars(symbol, timeframe, *, limit, rng=5.0):
    last_open = T0.replace(minute=0, second=0) - timedelta(hours=1)
    start = last_open - timedelta(hours=limit - 1)
    return [Bar(start + timedelta(hours=i), 100, 100 + rng / 2, 100 - rng / 2, 100, 1.0, symbol, "okx_demo") for i in range(limit)]


@pytest.fixture
def rig(tmp_path):
    cfg = QFConfig(state_dir=tmp_path)
    broker = FakeBroker()
    clock = Clock(T0)

    def quote(symbol):
        return Quote(symbol, broker.px - 0.01, broker.px, broker.px, clock(), "okx_demo")

    qf = QuickFlip(cfg, broker, get_bars=_bars, get_quote=quote, clock=clock, sleep=lambda s: None)
    return qf, broker, clock
