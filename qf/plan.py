"""Pure trade-planning math: volatility read, target/stop sizing, fee floor,
and price levels. No I/O -- every function here is a deterministic
transform of its arguments, so the numbers that decide whether a trade is
worth taking are unit-testable to the cent.

The one idea that shapes this module (Strategy_Spec §3): the spec's
"~2:1 reward:risk" is a GROSS ratio. Round-trip drag (two taker fees plus
slippage, ~0.9% on OKX demo) is subtracted from the reward and added to
the risk, so a small ATR-scaled target can be 2:1 gross and still need an
80%+ win rate to break even. `TradePlan.passes_fee_floor` makes that
visible before any order is placed.

All fractions are plain ratios (0.02 == 2%). Everything is long-only:
OKX spot, same as VibeTrading's crypto leg.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Sequence

from qf.config import QFConfig
from vt.data.feed import Bar
from vt.indicators.engine import atr as wilder_atr

#: Bar spacing may deviate this much from nominal before the read is
#: rejected (exchange gaps, DST-free UTC, etc. never come close).
_SPACING_TOLERANCE = 0.1
#: Precision (in units of one step) below which a value is float noise.
_UNIT_NOISE = Decimal("1e-9")


class PlanError(ValueError):
    """Inputs cannot produce a meaningful plan. Raised, never defaulted --
    a silently-zero ATR would size a zero-width stop."""


# --------------------------------------------------------------------------- #
# Cost model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeeModel:
    """Taker fee (both legs are market orders) plus a per-side slippage
    allowance. `taker_rate` should come from the live fee-rate read, with
    `QFConfig.fallback_taker_rate` only as a fallback."""

    taker_rate: float
    slippage_frac_per_side: float

    @property
    def round_trip_drag(self) -> float:
        return 2.0 * (self.taker_rate + self.slippage_frac_per_side)

    @property
    def breakeven_multiplier(self) -> float:
        """Exit price / entry price at which net P&L is exactly zero. The
        buy fee is taken from the base received and the sell fee from the
        quote proceeds, so the two legs compound rather than add."""
        return 1.0 / (1.0 - self.taker_rate) ** 2


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Levels:
    """Absolute prices for one position, derived from its actual entry."""

    entry_px: float
    target_px: float
    stop_px: float
    breakeven_px: float
    arm_px: float


@dataclass(frozen=True)
class TradePlan:
    symbol: str
    lane: str
    notional_usd: float
    atr_frac: float
    target_frac: float
    stop_frac: float
    drag_frac: float
    net_reward_frac: float
    net_risk_frac: float
    net_rr: float
    breakeven_win_rate: float
    min_net_rr: float
    passes_fee_floor: bool
    taker_rate: float
    arm_progress: float
    arm_min_cushion_frac: float
    max_hold_hours: float

    @property
    def planned_risk_usd(self) -> float:
        """Dollars lost if the stop fills at its trigger, fees included --
        the denominator of every R-multiple this project journals."""
        return self.notional_usd * self.net_risk_frac

    def levels(self, entry_px: float, *, tick_sz: float) -> Levels:
        """Absolute levels from the real fill price. Every rounding step
        errs toward LESS risk: the stop rounds up (tighter), breakeven and
        the arm trigger round up (never arm into a net loss), and the
        target rounds up (never book less than planned)."""
        if entry_px <= 0:
            raise PlanError(f"entry price must be positive, got {entry_px}")
        breakeven = round_up_to_step(entry_px / (1.0 - self.taker_rate) ** 2, tick_sz)
        halfway = entry_px * (1.0 + self.arm_progress * self.target_frac)
        cushioned = breakeven * (1.0 + self.arm_min_cushion_frac)
        return Levels(
            entry_px=entry_px,
            target_px=round_up_to_step(entry_px * (1.0 + self.target_frac), tick_sz),
            stop_px=round_up_to_step(entry_px * (1.0 - self.stop_frac), tick_sz),
            breakeven_px=breakeven,
            arm_px=round_up_to_step(max(halfway, cushioned), tick_sz),
        )


def build_plan(
    symbol: str,
    lane: str,
    notional_usd: float,
    *,
    atr: float,
    fees: FeeModel,
    cfg: QFConfig,
) -> TradePlan:
    """Size target and stop off the ATR read and score the result against
    the fee floor. `atr` is ATR1h as a fraction of price (`atr_frac`)."""
    if atr <= 0:
        raise PlanError(f"ATR fraction must be positive, got {atr}")
    if notional_usd <= 0:
        raise PlanError(f"notional must be positive, got {notional_usd}")

    target = cfg.k_atr * atr
    stop = cfg.stop_ratio * target
    drag = fees.round_trip_drag
    net_reward = target - drag
    net_risk = stop + drag
    net_rr = net_reward / net_risk
    breakeven_wr = net_risk / (net_reward + net_risk) if net_reward > 0 else 1.0

    return TradePlan(
        symbol=symbol,
        lane=lane,
        notional_usd=notional_usd,
        atr_frac=atr,
        target_frac=target,
        stop_frac=stop,
        drag_frac=drag,
        net_reward_frac=net_reward,
        net_risk_frac=net_risk,
        net_rr=net_rr,
        breakeven_win_rate=breakeven_wr,
        min_net_rr=cfg.min_net_rr,
        passes_fee_floor=net_rr >= cfg.min_net_rr,
        taker_rate=fees.taker_rate,
        arm_progress=cfg.arm_progress,
        arm_min_cushion_frac=cfg.arm_min_cushion_frac,
        max_hold_hours=cfg.max_hold_hours,
    )


# --------------------------------------------------------------------------- #
# Volatility read
# --------------------------------------------------------------------------- #


def atr_frac(
    bars: Sequence[Bar],
    *,
    period: int,
    now: datetime,
    bar_seconds: float = 3600.0,
) -> float:
    """Latest Wilder ATR over COMPLETED bars, as a fraction of the last
    completed close. The still-forming bar (opened less than `bar_seconds`
    ago) is dropped -- its partial range would understate volatility.

    Raises PlanError on too few bars or on bars whose spacing isn't
    `bar_seconds` (upstream silently maps an unknown timeframe token to
    daily bars, which would otherwise read as a 5x-too-wide target)."""
    completed = [b for b in bars if b.time + timedelta(seconds=bar_seconds) <= now]
    if len(completed) <= period + 1:
        raise PlanError(f"need more than {period + 1} completed bars for ATR({period}), got {len(completed)}")

    spacing = (completed[-1].time - completed[-2].time).total_seconds()
    if abs(spacing - bar_seconds) > bar_seconds * _SPACING_TOLERANCE:
        raise PlanError(f"bar spacing is {spacing:.0f}s, expected {bar_seconds:.0f}s -- wrong timeframe?")

    latest = wilder_atr(completed, period)[-1]
    close = completed[-1].close
    if latest is None or close <= 0:
        raise PlanError("ATR read produced no value")
    return latest / close


# --------------------------------------------------------------------------- #
# Exchange step rounding (Decimal, so str() of the result is exact)
# --------------------------------------------------------------------------- #


def round_up_to_step(value: float, step: float) -> float:
    return _round_to_step(value, step, ROUND_CEILING)


def round_down_to_step(value: float, step: float) -> float:
    return _round_to_step(value, step, ROUND_FLOOR)


def _round_to_step(value: float, step: float, rounding: str) -> float:
    if step <= 0:
        raise PlanError(f"step must be positive, got {step}")
    d_step = Decimal(repr(step))
    # Snap float noise first: 100 * 1.10 is 110.00000000000001, which a
    # bare ceiling would push a whole tick higher.
    exact_units = (Decimal(repr(value)) / d_step).quantize(_UNIT_NOISE)
    return float(exact_units.to_integral_value(rounding=rounding) * d_step)
