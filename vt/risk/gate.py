"""M006 -- Risk & Position Sizer. STUB ONLY (Phase C1) -- every function
below defines its real, final interface and raises NotImplementedError.
Implementation is Phase C2, not started yet; see `16_Next_Steps.md`
Phase C. Highest test priority in the project (T010-T014) -- tests are
written and confirmed failing against this stub *before* any real logic
lands, per this phase's TDD-mandatory rule.

Enforces `Risk_Policy.md` in full: computes size, places the stop, runs
the 12-step pre-trade gate, owns the circuit breakers. Net-new, and must
sit IN FRONT OF the upstream pre-trade path, never replace it (AD001).

Two invariants matter more than anything else in this module, and both
have property-based tests (T010, T011) rather than just examples:
  - No input should ever produce a size exceeding caps, divide by zero,
    or hand a NaN to an order.
  - No stop modification should ever increase distance-to-stop, for any
    caller, under any circumstance. This is the single most important
    assertion in the codebase.

Full contract in `03_Modules.md` section M006.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

RISK_PCT = 0.005  # Risk_Policy.md Sec1: 0.50% of current equity, paper phase
MAX_POSITION_PCT = 0.20  # Sec1: max position value, 20% of equity
MAX_CONCURRENT_POSITIONS = 3  # Sec1
MAX_SECTOR_POSITIONS = 2  # Sec1, equities only
STOP_ATR_MULTIPLIER = 1.5  # Sec2: initial stop = 1.5 x ATR(14, 5-min)
MAX_QUOTE_AGE_SECONDS = 5.0  # Sec5 step 7 / Risk_Policy.md data-staleness breaker
STOP_DISTANCE_MIN_PCT = 0.005  # Sec5 step 10: 0.5%-5% sane range
STOP_DISTANCE_MAX_PCT = 0.05
RUBRIC_MIN_SCORE = 9  # Sec5 step 11
DAILY_LOSS_LIMIT_R = -2.0  # Sec3
WEEKLY_LOSS_LIMIT_R = -6.0  # Sec3
DAILY_TRADE_CAP = 8  # Sec3
CONSECUTIVE_LOSS_HALT = 6  # Sec3
DRAWDOWN_KILL_R = -15.0  # Sec3

# Risk_Policy.md Sec5, in exact documented order -- T013 tests this sequence.
GATE_STEPS: tuple[str, ...] = (
    "kill_switch",
    "daily_weekly_loss_limit",
    "trade_cap",
    "calendar_stand_down",
    "max_concurrent_positions",
    "sector_correlation_cap",
    "quote_freshness",
    "broker_reconciliation",
    "size_within_caps",
    "stop_distance_sane",
    "rubric_threshold",
    "trade_card_written",
)


@dataclass(frozen=True)
class Signal:
    """The minimal shape the gate needs from upstream modules (M002
    candidate + M003 indicators + M005 rubric score) -- not the full
    downstream dataclasses, which don't exist yet either. This will very
    likely be replaced by a real composition of M002's `Candidate` and
    M005's `Score` once those modules reach this point in the build
    order; kept minimal and explicit here so M006's own tests aren't
    blocked on modules later in the critical path.
    """

    symbol: str
    side: Literal["long", "short"]
    entry_price: float
    atr: float
    calendar_multiplier: float  # 0.0 / 0.5 / 1.0, from M004 (or test-injected)
    rubric_score: int  # 0-12
    rubric_r1: int  # trend-alignment component, 0-2 -- gate requires >= 1
    rubric_r6: int  # sixth rubric component, 0-2 -- gate requires >= 1
    quote_age_seconds: float
    open_positions: int
    sector_positions: int
    broker_reconciled: bool
    card_written: bool


@dataclass(frozen=True)
class Decision:
    status: Literal["approved", "rejected", "halted"]
    size: float
    stop_price: float | None
    reject_reason: str | None  # one of GATE_STEPS, or None when approved


@dataclass
class BreakerState:
    """Circuit breaker state. Must be persisted (T012: 'survive a process
    restart') -- an in-memory-only implementation is treated as a defect,
    not an acceptable simplification.
    """

    session_date: str | None = None  # ISO date; a new date resets session counters
    session_r: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    week_start_date: str | None = None
    weekly_r: float = 0.0
    equity_peak: float = 0.0
    halted_until: datetime | None = None
    hard_killed: bool = False


def compute_stop(*, side: Literal["long", "short"], entry_price: float, atr: float) -> float:
    """Risk_Policy.md Sec2: initial stop = 1.5 x ATR(14, 5-min) from entry."""
    raise NotImplementedError("M006 stub -- Phase C2, see 16_Next_Steps.md")


def compute_size(
    *,
    equity: float,
    entry_price: float,
    stop_price: float,
    calendar_multiplier: float,
    risk_pct: float = RISK_PCT,
    max_position_pct: float = MAX_POSITION_PCT,
) -> float:
    """Risk_Policy.md Sec1: fixed-fractional sizing.

    risk_per_trade = equity * risk_pct * calendar_multiplier
    size = risk_per_trade / abs(entry_price - stop_price), capped so
    notional (size * entry_price) never exceeds max_position_pct * equity.

    Must never raise -- including when entry_price == stop_price (zero
    stop distance) or equity == 0 -- and must never return a negative
    size or NaN (T010).
    """
    raise NotImplementedError("M006 stub -- Phase C2, see 16_Next_Steps.md")


def validate_stop_change(
    *, side: Literal["long", "short"], entry_price: float, old_stop: float, new_stop: float
) -> bool:
    """Risk_Policy.md Sec2: stops can be tightened, never widened.

    True iff `new_stop` does not increase distance-to-stop relative to
    `old_stop` (distance measured from `entry_price`). No caller, code
    path, or special case may bypass this (T011).
    """
    raise NotImplementedError("M006 stub -- Phase C2, see 16_Next_Steps.md")


def record_trade(state: BreakerState, *, r_multiple: float, equity: float, now: datetime) -> BreakerState:
    """Update breaker state after a closed trade and return a **new**
    state (never mutates `state` in place, per the project's immutability
    convention). Trips the breakers per Risk_Policy.md Sec3.
    """
    raise NotImplementedError("M006 stub -- Phase C2, see 16_Next_Steps.md")


def is_halted(state: BreakerState, *, now: datetime) -> bool:
    """Whether any breaker currently blocks new entries, evaluated as of
    `now` (session/weekly counters reset on a calendar boundary; a hard
    kill from the drawdown breaker never auto-clears -- Risk_Policy.md
    Sec3's 'Manual only' reset).
    """
    raise NotImplementedError("M006 stub -- Phase C2, see 16_Next_Steps.md")


def save_breaker_state(state: BreakerState, path: Path) -> None:
    """Persistence is not optional -- T012 requires breakers to survive a
    process restart.
    """
    raise NotImplementedError("M006 stub -- Phase C2, see 16_Next_Steps.md")


def load_breaker_state(path: Path) -> BreakerState:
    raise NotImplementedError("M006 stub -- Phase C2, see 16_Next_Steps.md")


def evaluate(signal: Signal, *, equity: float, breaker_state: BreakerState, now: datetime) -> Decision:
    """The 12-step pre-trade gate, Risk_Policy.md Sec5, in the exact
    documented order (`GATE_STEPS`). First failure short-circuits (T013)
    -- later steps are never evaluated once an earlier one fails. Step 8
    (broker reconciliation) produces `status="halted"`, not `"rejected"`
    -- a state-drift halt is a different severity than an ordinary reject.
    A signal whose Trade Card was never journaled is rejected at step 12
    even if every earlier step passed.
    """
    raise NotImplementedError("M006 stub -- Phase C2, see 16_Next_Steps.md")


def kill() -> None:
    """Standalone, reachable without the UI (Risk_Policy.md Sec4) --
    flattens everything and sets a hard-kill breaker state. See also
    T023 / `vt/alerts/kill.py` (Phase C4), which must work even if this
    process itself is wedged.
    """
    raise NotImplementedError("M006 stub -- Phase C2, see 16_Next_Steps.md")
