"""Tests for qf.plan -- the pure target/stop/fee-floor math.

Every number here is hand-checkable: taker 0.35% + 0.10% slippage per side
gives a 0.90% round-trip drag, which is the figure the whole fee-floor
argument in `QuickFlip/Strategy_Spec.md` §3 rests on.

Run with: pytest qf/tests -m unit
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from qf.config import QFConfig
from qf.plan import (
    FeeModel,
    PlanError,
    atr_frac,
    build_plan,
    round_down_to_step,
    round_up_to_step,
)
from vt.data.feed import Bar

pytestmark = pytest.mark.unit

FEES = FeeModel(taker_rate=0.0035, slippage_frac_per_side=0.001)
CFG = QFConfig()
NOW = datetime(2026, 9, 25, 12, 30, tzinfo=timezone.utc)


def _hourly_bars(n: int, *, last_open: datetime, close: float = 100.0, rng: float = 2.0) -> list[Bar]:
    """n hourly bars ending with one that opened at `last_open`, every bar
    spanning exactly `rng` around a flat close, so Wilder ATR == rng."""
    start = last_open - timedelta(hours=n - 1)
    return [
        Bar(
            time=start + timedelta(hours=i),
            open=close,
            high=close + rng / 2,
            low=close - rng / 2,
            close=close,
            volume=1.0,
            symbol="SOL-USDT",
            source_feed="okx_demo",
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #
# Cost model
# --------------------------------------------------------------------------- #


def test_round_trip_drag_is_two_sides_of_fee_plus_slippage():
    assert FEES.round_trip_drag == pytest.approx(0.009)


def test_breakeven_multiplier_compounds_both_fee_legs():
    # buy fee comes out of the base received, sell fee out of the proceeds
    assert FEES.breakeven_multiplier == pytest.approx(1 / (0.9965**2))


# --------------------------------------------------------------------------- #
# build_plan -- the fee floor is the headline behaviour
# --------------------------------------------------------------------------- #


def test_calm_regime_fails_fee_floor_even_though_gross_is_two_to_one():
    plan = build_plan("BTC-USDT", "small", 1_000.0, atr=0.01, fees=FEES, cfg=CFG)

    assert plan.target_frac == pytest.approx(0.02)
    assert plan.stop_frac == pytest.approx(0.01)
    assert plan.net_reward_frac == pytest.approx(0.011)
    assert plan.net_risk_frac == pytest.approx(0.019)
    assert plan.net_rr == pytest.approx(0.011 / 0.019)
    assert plan.breakeven_win_rate == pytest.approx(0.019 / 0.030)
    assert plan.passes_fee_floor is False


def test_volatile_regime_clears_fee_floor():
    plan = build_plan("NEAR-USDT", "small", 1_000.0, atr=0.025, fees=FEES, cfg=CFG)

    assert plan.target_frac == pytest.approx(0.05)
    assert plan.net_rr == pytest.approx(0.041 / 0.034)
    assert plan.breakeven_win_rate < 0.5
    assert plan.passes_fee_floor is True


def test_target_that_cannot_cover_drag_reports_certain_loss():
    plan = build_plan("BTC-USDT", "small", 1_000.0, atr=0.002, fees=FEES, cfg=CFG)

    assert plan.net_reward_frac < 0
    assert plan.breakeven_win_rate == 1.0
    assert plan.passes_fee_floor is False


def test_planned_risk_usd_is_notional_times_net_risk():
    plan = build_plan("SOL-USDT", "large", 4_000.0, atr=0.01, fees=FEES, cfg=CFG)
    assert plan.planned_risk_usd == pytest.approx(4_000.0 * 0.019)


@pytest.mark.parametrize("bad_atr", [0.0, -0.01])
def test_non_positive_atr_is_refused(bad_atr):
    with pytest.raises(PlanError):
        build_plan("SOL-USDT", "small", 1_000.0, atr=bad_atr, fees=FEES, cfg=CFG)


def test_non_positive_notional_is_refused():
    with pytest.raises(PlanError):
        build_plan("SOL-USDT", "small", 0.0, atr=0.01, fees=FEES, cfg=CFG)


# --------------------------------------------------------------------------- #
# Levels -- rounding always errs toward less risk
# --------------------------------------------------------------------------- #


def test_levels_from_a_clean_entry():
    plan = build_plan("SOL-USDT", "small", 1_000.0, atr=0.01, fees=FEES, cfg=CFG)
    lv = plan.levels(100.0, tick_sz=0.01)

    assert lv.target_px == 102.0
    assert lv.stop_px == 99.0
    assert lv.breakeven_px == 100.71  # 100 / 0.9965^2 = 100.7026 -> ceil tick
    assert lv.arm_px == 101.0  # halfway to target beats breakeven + cushion


def test_stop_rounds_up_so_risk_never_exceeds_plan():
    plan = build_plan("SOL-USDT", "small", 1_000.0, atr=0.01, fees=FEES, cfg=CFG)
    lv = plan.levels(100.555, tick_sz=0.01)

    raw_stop = 100.555 * 0.99
    assert lv.stop_px >= raw_stop
    assert lv.stop_px - raw_stop < 0.01
    assert lv.stop_px == 99.55


def test_arm_falls_back_to_breakeven_cushion_when_target_is_small():
    plan = build_plan("BTC-USDT", "small", 1_000.0, atr=0.002, fees=FEES, cfg=CFG)
    lv = plan.levels(100.0, tick_sz=0.01)

    # halfway-to-target (100.20) sits BELOW net breakeven, so arming there
    # would put the stop above the price; cushion keeps it tradeable
    assert lv.arm_px == 100.92  # 100.71 * 1.002 = 100.911 -> ceil tick
    assert lv.arm_px > lv.breakeven_px


def test_levels_refuse_non_positive_entry():
    plan = build_plan("SOL-USDT", "small", 1_000.0, atr=0.01, fees=FEES, cfg=CFG)
    with pytest.raises(PlanError):
        plan.levels(0.0, tick_sz=0.01)


# --------------------------------------------------------------------------- #
# ATR read
# --------------------------------------------------------------------------- #


def test_atr_frac_on_constant_range_bars():
    bars = _hourly_bars(40, last_open=NOW - timedelta(hours=2))
    assert atr_frac(bars, period=14, now=NOW) == pytest.approx(0.02)


def test_atr_frac_drops_the_still_forming_bar():
    bars = _hourly_bars(40, last_open=NOW - timedelta(minutes=10))
    # make the forming bar wildly wide -- it must not leak into the read
    forming = bars[-1]
    bars[-1] = Bar(forming.time, 100.0, 150.0, 50.0, 100.0, 1.0, forming.symbol, forming.source_feed)
    assert atr_frac(bars, period=14, now=NOW) == pytest.approx(0.02)


def test_atr_frac_refuses_too_few_bars():
    bars = _hourly_bars(10, last_open=NOW - timedelta(hours=2))
    with pytest.raises(PlanError, match="need"):
        atr_frac(bars, period=14, now=NOW)


def test_atr_frac_refuses_wrong_bar_spacing():
    # upstream maps an unknown timeframe token to daily bars silently
    start = NOW - timedelta(days=40)
    daily = [
        Bar(start + timedelta(days=i), 100, 101, 99, 100, 1.0, "SOL-USDT", "okx_demo") for i in range(40)
    ]
    with pytest.raises(PlanError, match="spacing"):
        atr_frac(daily, period=14, now=NOW)


# --------------------------------------------------------------------------- #
# Step rounding
# --------------------------------------------------------------------------- #


def test_round_up_to_step():
    assert round_up_to_step(1.0000001, 0.01) == 1.01
    assert round_up_to_step(1.23, 0.01) == 1.23


def test_round_up_ignores_float_noise():
    # 100 * 1.10 == 110.00000000000001 in binary floating point
    assert round_up_to_step(100 * 1.10, 0.01) == 110.0
    assert round_down_to_step(0.1 + 0.2, 0.1) == 0.3


def test_round_down_to_step():
    assert round_down_to_step(0.123456789, 0.000001) == 0.123456
    assert round_down_to_step(5.0, 0.001) == 5.0
