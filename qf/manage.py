"""Pure exit management: what should happen to an open position right now.

Target and stop are NOT decided here -- they live on the exchange as one
OCO order, so they fire even if this process is dead. The monitor only
owns the two exits the exchange can't express on its own:

  * the breakeven ARM -- once price covers `arm_progress` of the way to
    target, raise the stop to net breakeven (never lower it: stops only
    move toward price, `Risk_Policy.md` §1);
  * the TIME_EXIT -- flatten at the 8-hour deadline whatever the P&L.

Plus the P&L / outcome arithmetic for the journal, kept here so it is
unit-tested apart from any broker plumbing.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from enum import Enum
from typing import Any

from qf.state import Position


class Action(Enum):
    HOLD = "hold"
    ARM = "arm"
    TIME_EXIT = "time_exit"


def observe(pos: Position, last_px: float) -> Position:
    """Fold one price observation into the position's high/low watermarks
    (for MFE/MAE in the journal). Returns a new Position."""
    high = max(pos.high_px, last_px) if pos.high_px else last_px
    low = min(pos.low_px, last_px) if pos.low_px else last_px
    return replace(pos, high_px=high, low_px=low)


def decide(pos: Position, last_px: float, now: datetime) -> Action:
    """Time-stop outranks everything: past the deadline there is nothing
    left to manage, only to close."""
    if now >= pos.deadline:
        return Action.TIME_EXIT
    if not pos.armed and last_px >= pos.arm_px:
        return Action.ARM
    return Action.HOLD


def armed_stop(pos: Position) -> float:
    """The stop the ARM moves to. `max` enforces never-loosen even if a
    later tweak ever put breakeven below the original stop."""
    return max(pos.stop_px, pos.breakeven_px)


def realized_pnl(pos: Position, *, exit_px: float, exit_size: float, exit_fee_quote: float) -> float:
    """Net quote-currency P&L. `notional_usd` is what the entry cost in
    quote (its fee came out of the base received, already reflected in the
    smaller `size`); the exit fee is charged in quote. Any lot-rounding
    dust left behind is counted as a cost, which is what it is."""
    return exit_px * exit_size - exit_fee_quote - pos.notional_usd


def outcome(
    pos: Position,
    *,
    exit_reason: str,
    exit_px: float,
    exit_size: float,
    exit_fee_quote: float,
    closed_at: datetime,
) -> dict[str, Any]:
    """Journal outcome row. `r_multiple` is net P&L over the net risk the
    plan accepted at entry, so +1R means "won as much as the stop would
    have cost", fees included on both sides of the ratio."""
    pnl = realized_pnl(pos, exit_px=exit_px, exit_size=exit_size, exit_fee_quote=exit_fee_quote)
    held_min = (closed_at - pos.opened_at).total_seconds() / 60.0
    return {
        "r_multiple": pnl / pos.planned_risk_usd if pos.planned_risk_usd > 0 else 0.0,
        "exit_reason": exit_reason,
        "pnl_usd": pnl,
        "pnl_frac": pnl / pos.notional_usd if pos.notional_usd > 0 else 0.0,
        "entry_px": pos.entry_px,
        "exit_px": exit_px,
        "exit_size": exit_size,
        "exit_fee_quote": exit_fee_quote,
        "armed": pos.armed,
        "manual": pos.manual,
        "mfe_frac": (pos.high_px / pos.entry_px - 1.0) if pos.high_px else None,
        "mae_frac": (pos.low_px / pos.entry_px - 1.0) if pos.low_px else None,
        "time_in_trade_minutes": held_min,
    }
