"""M010 -- Backtest cost model (T021). Phase E.

Every backtest fill must carry commission + spread + modeled slippage.
A backtest run under an all-zero cost model reports fantasy fills --
`Metrics_Definitions.md`'s "Cost ratio" metric and the B-tier acceptance
criterion ("costs modeled at >= 2x observed slippage and still
positive") both presume costs were modeled in the first place. This
module is the one place that presumption is actually enforced: a
walk-forward run cannot proceed under a `CostModel` whose three
components are all zero (`ZeroCostBacktestError`).

Two preset `CostModel`s are provided as a documented starting point.
`CRYPTO_DEFAULT.commission_bps` is now REAL, observed data (S030): three
real OKX fills -- a manual mechanics test, not a strategy signal --
each charged exactly 0.35% taker fee, replacing the original 8bps
literature guess. Its `spread_bps`/`slippage_bps`, and all of
`EQUITIES_DEFAULT`, remain unvalidated literature-standard assumptions,
stated as such where each constant is defined below.

Full contract in `03_Modules.md` section M010; test spec in
`06_Tests.md` T021.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


class ZeroCostBacktestError(RuntimeError):
    """Raised by `validate_cost_model` when a `CostModel` has all three
    components at zero. A zero-cost backtest is not a stricter test --
    it silently overstates every fill, which is exactly the kind of
    result `Metrics_Definitions.md` § 3 and the B-tier "costs modeled at
    >= 2x observed slippage" criterion exist to catch.
    """


@dataclass(frozen=True)
class CostModel:
    """All three components are expressed in basis points (bps) of
    fill notional (`price * quantity`). `commission_bps` is a broker
    fee; `spread_bps` and `slippage_bps` are both modeled market-impact
    costs but kept separate so a caller can report them individually
    (`Metrics_Definitions.md`'s "Cost ratio" wants fees+spread; the
    "Slippage" metric wants the ATR-normalized fill delta on its own).
    """

    commission_bps: float
    spread_bps: float
    slippage_bps: float


@dataclass(frozen=True)
class AdjustedFill:
    """The result of applying a `CostModel` to one raw fill price.
    `effective_price` is what the walk-forward harness should actually
    use for P&L -- always worse than `raw_price` for the trader, in the
    direction that matches `side` (buy pays more, sell receives less).
    """

    raw_price: float
    effective_price: float
    side: Literal["buy", "sell"]
    quantity: float
    commission: float
    spread_cost: float
    slippage_cost: float
    total_cost: float
    total_cost_bps: float


def validate_cost_model(model: CostModel) -> None:
    for name, value in (
        ("commission_bps", model.commission_bps),
        ("spread_bps", model.spread_bps),
        ("slippage_bps", model.slippage_bps),
    ):
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    if model.commission_bps == 0 and model.spread_bps == 0 and model.slippage_bps == 0:
        raise ZeroCostBacktestError(
            "CostModel has commission_bps=spread_bps=slippage_bps=0 -- a backtest "
            "run under this model reports fantasy fills (T021, Metrics_Definitions.md "
            "§ 3, B-tier 'costs modeled at >= 2x observed slippage'). Supply at least "
            "one nonzero component, or use EQUITIES_DEFAULT / CRYPTO_DEFAULT."
        )


def apply_costs(*, raw_price: float, side: Literal["buy", "sell"], quantity: float, model: CostModel) -> AdjustedFill:
    if raw_price <= 0:
        raise ValueError(f"raw_price must be > 0, got {raw_price}")
    if quantity <= 0:
        raise ValueError(f"quantity must be > 0, got {quantity}")
    validate_cost_model(model)

    notional = raw_price * quantity
    commission = notional * model.commission_bps / 10_000
    spread_cost = notional * model.spread_bps / 10_000
    slippage_cost = notional * model.slippage_bps / 10_000
    total_cost = commission + spread_cost + slippage_cost
    total_cost_bps = model.commission_bps + model.spread_bps + model.slippage_bps

    cost_per_unit = total_cost / quantity
    effective_price = raw_price + cost_per_unit if side == "buy" else raw_price - cost_per_unit

    return AdjustedFill(
        raw_price=raw_price,
        effective_price=effective_price,
        side=side,
        quantity=quantity,
        commission=commission,
        spread_cost=spread_cost,
        slippage_cost=slippage_cost,
        total_cost=total_cost,
        total_cost_bps=total_cost_bps,
    )


#: Starting assumptions, not yet validated against real fills (see module
#: docstring). Alpaca equities trading itself is commission-free; spread
#: and slippage still apply even on liquid names.
EQUITIES_DEFAULT = CostModel(commission_bps=0.0, spread_bps=1.5, slippage_bps=5.0)

#: commission_bps=35.0 is REAL, observed OKX demo-account taker fee data
#: (S030): three real fills -- BTC-USDT, ETH-USDT, SOL-USDT, all
#: marketable orders -- each charged exactly 0.35% (feeRate="-0.0035" on
#: every fill in `get_fills_history`), 4x+ the original 8bps literature
#: guess this constant shipped with. spread_bps/slippage_bps are still
#: unvalidated estimates -- three same-minute round trips on a quiet
#: tape can't isolate spread/slippage from the fee, only the fee itself
#: was directly observable.
CRYPTO_DEFAULT = CostModel(commission_bps=35.0, spread_bps=2.0, slippage_bps=5.0)
