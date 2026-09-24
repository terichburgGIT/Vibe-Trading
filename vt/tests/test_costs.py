"""Tests for M010 -- vt.validate.costs (T021; see 06_Tests.md).

Run with: pytest vt/tests -m unit
"""

from __future__ import annotations

import pytest

from vt.validate import costs

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# T021 -- zero-cost backtests rejected
# --------------------------------------------------------------------------- #


def test_validate_cost_model_rejects_all_zero() -> None:
    """The exact scenario T021 exists to catch: every component zero."""
    model = costs.CostModel(commission_bps=0.0, spread_bps=0.0, slippage_bps=0.0)
    with pytest.raises(costs.ZeroCostBacktestError):
        costs.validate_cost_model(model)


def test_validate_cost_model_accepts_commission_only() -> None:
    """Any single nonzero component is enough -- T021 doesn't require all three,
    just not all zero (some venues are genuinely commission-free, e.g. Alpaca)."""
    model = costs.CostModel(commission_bps=0.0, spread_bps=1.5, slippage_bps=0.0)
    costs.validate_cost_model(model)  # must not raise


def test_validate_cost_model_rejects_negative_component() -> None:
    """A negative cost isn't 'generous', it's a modeling bug (backwards sign)."""
    model = costs.CostModel(commission_bps=-1.0, spread_bps=1.0, slippage_bps=1.0)
    with pytest.raises(ValueError):
        costs.validate_cost_model(model)


def test_equities_default_and_crypto_default_are_not_zero_cost() -> None:
    """The two shipped presets must themselves satisfy T021 -- a preset that
    fails its own validator would be a silent self-contradiction."""
    costs.validate_cost_model(costs.EQUITIES_DEFAULT)
    costs.validate_cost_model(costs.CRYPTO_DEFAULT)


# --------------------------------------------------------------------------- #
# T021 -- every fill carries commission + spread + slippage
# --------------------------------------------------------------------------- #


def test_apply_costs_buy_fill_costs_more_than_raw_price() -> None:
    model = costs.CostModel(commission_bps=10.0, spread_bps=5.0, slippage_bps=5.0)
    fill = costs.apply_costs(raw_price=100.0, side="buy", quantity=10.0, model=model)

    assert fill.effective_price > fill.raw_price
    assert fill.total_cost_bps == pytest.approx(20.0)


def test_apply_costs_sell_fill_receives_less_than_raw_price() -> None:
    model = costs.CostModel(commission_bps=10.0, spread_bps=5.0, slippage_bps=5.0)
    fill = costs.apply_costs(raw_price=100.0, side="sell", quantity=10.0, model=model)

    assert fill.effective_price < fill.raw_price


def test_apply_costs_total_cost_matches_sum_of_components() -> None:
    """Hand-computed: 100.0 * 10 qty = $1000 notional. 20bps total = $2.00."""
    model = costs.CostModel(commission_bps=10.0, spread_bps=5.0, slippage_bps=5.0)
    fill = costs.apply_costs(raw_price=100.0, side="buy", quantity=10.0, model=model)

    assert fill.commission == pytest.approx(1.00)  # 10bps of $1000
    assert fill.spread_cost == pytest.approx(0.50)  # 5bps of $1000
    assert fill.slippage_cost == pytest.approx(0.50)  # 5bps of $1000
    assert fill.total_cost == pytest.approx(2.00)
    assert fill.effective_price == pytest.approx(100.0 + 2.00 / 10.0)  # cost spread over quantity


def test_apply_costs_symmetric_round_trip_always_loses_money() -> None:
    """Buy then immediately sell at the SAME raw price must show a net loss
    equal to twice the one-way cost -- costs must never net out to zero or
    (worse) turn into a profit from rounding."""
    model = costs.CostModel(commission_bps=10.0, spread_bps=5.0, slippage_bps=5.0)
    buy = costs.apply_costs(raw_price=50.0, side="buy", quantity=4.0, model=model)
    sell = costs.apply_costs(raw_price=50.0, side="sell", quantity=4.0, model=model)

    round_trip_pnl = (sell.effective_price - buy.effective_price) * 4.0
    assert round_trip_pnl < 0
    assert round_trip_pnl == pytest.approx(-(buy.total_cost + sell.total_cost))


def test_apply_costs_rejects_zero_cost_model() -> None:
    """apply_costs must not silently accept the exact model validate_cost_model
    rejects -- the guard has to actually sit in the fill path, not just exist
    as a function nobody calls."""
    model = costs.CostModel(commission_bps=0.0, spread_bps=0.0, slippage_bps=0.0)
    with pytest.raises(costs.ZeroCostBacktestError):
        costs.apply_costs(raw_price=100.0, side="buy", quantity=1.0, model=model)


def test_apply_costs_rejects_nonpositive_price_or_quantity() -> None:
    model = costs.CostModel(commission_bps=10.0, spread_bps=5.0, slippage_bps=5.0)
    with pytest.raises(ValueError):
        costs.apply_costs(raw_price=0.0, side="buy", quantity=1.0, model=model)
    with pytest.raises(ValueError):
        costs.apply_costs(raw_price=100.0, side="buy", quantity=0.0, model=model)
    with pytest.raises(ValueError):
        costs.apply_costs(raw_price=100.0, side="buy", quantity=-1.0, model=model)
