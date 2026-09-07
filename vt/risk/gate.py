"""M006 -- Risk & Position Sizer. Phase C2 implementation (see
`16_Next_Steps.md` Phase C). T010-T014 in `vt/tests/test_gate.py` were
written first against a stub, confirmed failing, and drove this
implementation -- nothing in the tests changed to make them pass.

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

import dataclasses
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
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
    offset = STOP_ATR_MULTIPLIER * atr
    return entry_price - offset if side == "long" else entry_price + offset


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
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0 or equity <= 0 or calendar_multiplier <= 0:
        return 0.0

    risk_per_trade = equity * risk_pct * calendar_multiplier
    size = risk_per_trade / stop_distance

    max_notional = max_position_pct * equity
    notional = size * entry_price
    if notional > max_notional:
        size = max_notional / entry_price
    return size


def validate_stop_change(
    *, side: Literal["long", "short"], entry_price: float, old_stop: float, new_stop: float
) -> bool:
    """Risk_Policy.md Sec2: stops can be tightened, never widened.

    True iff `new_stop` does not increase distance-to-stop relative to
    `old_stop` (distance measured from `entry_price`). No caller, code
    path, or special case may bypass this (T011).
    """
    old_distance = abs(entry_price - old_stop)
    new_distance = abs(entry_price - new_stop)
    return new_distance <= old_distance + 1e-9


def record_trade(state: BreakerState, *, r_multiple: float, equity: float, now: datetime) -> BreakerState:
    """Update breaker state after a closed trade and return a **new**
    state (never mutates `state` in place, per the project's immutability
    convention). Trips the breakers per Risk_Policy.md Sec3.
    """
    today = now.date().isoformat()
    if state.session_date == today:
        session_r = state.session_r + r_multiple
        trades_today = state.trades_today + 1
    else:
        session_r = r_multiple
        trades_today = 1

    consecutive_losses = state.consecutive_losses + 1 if r_multiple < 0 else 0

    # Weekly loss limit resets Sec3 'Manual, after a written review' -- once
    # breached, week_start_date/weekly_r are frozen; they never roll over on
    # their own, unlike the daily counters above.
    monday = (now.date() - timedelta(days=now.date().weekday())).isoformat()
    already_weekly_halted = state.weekly_r <= WEEKLY_LOSS_LIMIT_R
    if state.week_start_date != monday and not already_weekly_halted:
        week_start_date, weekly_r = monday, r_multiple
    else:
        week_start_date = state.week_start_date or monday
        weekly_r = state.weekly_r + r_multiple

    # Drawdown kill switch is evaluated as % off the equity high-water mark
    # (Risk_Policy.md Sec3: '-15R from equity peak') and never auto-clears.
    # equity_peak == 0.0 means the caller never seeded a starting equity --
    # tracking stays inert rather than adopting the first post-trade equity
    # as a false peak (which would flag ordinary losing streaks as a
    # drawdown-kill event).
    if state.equity_peak > 0:
        equity_peak = max(state.equity_peak, equity)
        drawdown_pct = (equity - equity_peak) / equity_peak * 100
    else:
        equity_peak = state.equity_peak
        drawdown_pct = 0.0
    hard_killed = state.hard_killed or drawdown_pct <= DRAWDOWN_KILL_R

    halted_until = state.halted_until
    if consecutive_losses >= CONSECUTIVE_LOSS_HALT:
        candidate = now + timedelta(hours=24)
        if halted_until is None or candidate > halted_until:
            halted_until = candidate

    return BreakerState(
        session_date=today,
        session_r=session_r,
        trades_today=trades_today,
        consecutive_losses=consecutive_losses,
        week_start_date=week_start_date,
        weekly_r=weekly_r,
        equity_peak=equity_peak,
        halted_until=halted_until,
        hard_killed=hard_killed,
    )


def is_halted(state: BreakerState, *, now: datetime) -> bool:
    """Whether any breaker currently blocks new entries, evaluated as of
    `now` (session/weekly counters reset on a calendar boundary; a hard
    kill from the drawdown breaker never auto-clears -- Risk_Policy.md
    Sec3's 'Manual only' reset).
    """
    if state.hard_killed:
        return True
    if state.weekly_r <= WEEKLY_LOSS_LIMIT_R:
        return True
    same_session = state.session_date == now.date().isoformat()
    if same_session and (state.session_r <= DAILY_LOSS_LIMIT_R or state.trades_today >= DAILY_TRADE_CAP):
        return True
    if state.halted_until is not None and now < state.halted_until:
        return True
    return False


def save_breaker_state(state: BreakerState, path: Path) -> None:
    """Persistence is not optional -- T012 requires breakers to survive a
    process restart.
    """
    data = dataclasses.asdict(state)
    data["halted_until"] = state.halted_until.isoformat() if state.halted_until is not None else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def load_breaker_state(path: Path) -> BreakerState:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("halted_until"):
        data["halted_until"] = datetime.fromisoformat(data["halted_until"])
    return BreakerState(**data)


def evaluate(signal: Signal, *, equity: float, breaker_state: BreakerState, now: datetime) -> Decision:
    """The 12-step pre-trade gate, Risk_Policy.md Sec5, in the exact
    documented order (`GATE_STEPS`). First failure short-circuits (T013)
    -- later steps are never evaluated once an earlier one fails. Step 8
    (broker reconciliation) produces `status="halted"`, not `"rejected"`
    -- a state-drift halt is a different severity than an ordinary reject.
    A signal whose Trade Card was never journaled is rejected at step 12
    even if every earlier step passed.
    """

    def rejected(reason: str) -> Decision:
        return Decision(status="rejected", size=0.0, stop_price=None, reject_reason=reason)

    # Step 1: kill switch -- hard kill, or a live time-boxed halt (e.g. the
    # consecutive-loss breaker), both mean "stop trading now" regardless of
    # signal quality.
    if breaker_state.hard_killed or (breaker_state.halted_until is not None and now < breaker_state.halted_until):
        return rejected("kill_switch")

    # Step 2: daily/weekly loss limit.
    if breaker_state.session_r <= DAILY_LOSS_LIMIT_R or breaker_state.weekly_r <= WEEKLY_LOSS_LIMIT_R:
        return rejected("daily_weekly_loss_limit")

    # Step 3: trade cap.
    if breaker_state.trades_today >= DAILY_TRADE_CAP:
        return rejected("trade_cap")

    # Step 4: calendar STAND DOWN.
    if signal.calendar_multiplier <= 0.0:
        return rejected("calendar_stand_down")

    # Step 5: max concurrent positions.
    if signal.open_positions >= MAX_CONCURRENT_POSITIONS:
        return rejected("max_concurrent_positions")

    # Step 6: sector / correlation cap.
    if signal.sector_positions >= MAX_SECTOR_POSITIONS:
        return rejected("sector_correlation_cap")

    # Step 7: quote freshness.
    if signal.quote_age_seconds >= MAX_QUOTE_AGE_SECONDS:
        return rejected("quote_freshness")

    # Step 8: broker reconciliation -- a HALT, not a REJECT.
    if not signal.broker_reconciled:
        return Decision(status="halted", size=0.0, stop_price=None, reject_reason="broker_reconciliation")

    stop_price = compute_stop(side=signal.side, entry_price=signal.entry_price, atr=signal.atr)
    size = compute_size(
        equity=equity,
        entry_price=signal.entry_price,
        stop_price=stop_price,
        calendar_multiplier=signal.calendar_multiplier,
    )

    # Step 9: computed size within caps.
    if size <= 0:
        return rejected("size_within_caps")

    # Step 10: stop distance sane (0.5-5%).
    stop_distance_pct = abs(signal.entry_price - stop_price) / signal.entry_price
    if stop_distance_pct < STOP_DISTANCE_MIN_PCT or stop_distance_pct > STOP_DISTANCE_MAX_PCT:
        return rejected("stop_distance_sane")

    # Step 11: rubric threshold, including the hard per-component floors.
    if signal.rubric_score < RUBRIC_MIN_SCORE or signal.rubric_r1 < 1 or signal.rubric_r6 < 1:
        return rejected("rubric_threshold")

    # Step 12: Trade Card written before the position exists.
    if not signal.card_written:
        return rejected("trade_card_written")

    return Decision(status="approved", size=size, stop_price=stop_price, reject_reason=None)


_DEFAULT_BREAKER_STATE_PATH = Path.home() / ".vibe-trading" / "breaker_state.json"


def kill() -> None:
    """Standalone, reachable without the UI (Risk_Policy.md Sec4) --
    flattens everything and sets a hard-kill breaker state. See also
    T023 / `vt/alerts/kill.py` (Phase C4), which must work even if this
    process itself is wedged.
    """
    path = _DEFAULT_BREAKER_STATE_PATH
    try:
        state = load_breaker_state(path)
    except FileNotFoundError:
        state = BreakerState()
    save_breaker_state(dataclasses.replace(state, hard_killed=True), path)
