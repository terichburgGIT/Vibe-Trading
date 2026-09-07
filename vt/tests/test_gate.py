"""Tests for M006 -- vt.risk.gate (T010-T014; see 06_Tests.md).

Phase C1 (`16_Next_Steps.md`): these are written against a STUB -- every
`vt.risk.gate` function currently raises NotImplementedError. That is the
correct, expected failure for every test in this file right now. Phase C2
implements `gate.py` for real, one function at a time, until these go
green -- nothing here should need to change to make that happen; the
spec is written as if the implementation already existed.

T010/T011 are property-based (Hypothesis), per this phase's own rule
("property-based, not example-based") -- explicit degenerate-case tests
sit alongside each to guarantee ATR->0, equity->0, and price->0.01 are
always exercised regardless of what Hypothesis happens to generate.

Run with: pytest vt/tests -m unit
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vt.risk import gate

pytestmark = pytest.mark.unit

_FLOAT = st.floats(allow_nan=False, allow_infinity=False)


# --------------------------------------------------------------------------- #
# T010 -- Position size never exceeds caps (property test)
# --------------------------------------------------------------------------- #


@given(
    equity=st.floats(min_value=0, max_value=10_000_000, allow_nan=False, allow_infinity=False),
    entry_price=st.floats(min_value=0.01, max_value=100_000, allow_nan=False, allow_infinity=False),
    atr=st.floats(min_value=0, max_value=10_000, allow_nan=False, allow_infinity=False),
    calendar_multiplier=st.sampled_from([0.0, 0.5, 1.0]),
    side=st.sampled_from(["long", "short"]),
)
@settings(max_examples=200)
def test_size_never_exceeds_caps_or_produces_invalid_output(
    equity: float, entry_price: float, atr: float, calendar_multiplier: float, side: str
) -> None:
    stop_price = entry_price - gate.STOP_ATR_MULTIPLIER * atr if side == "long" else entry_price + gate.STOP_ATR_MULTIPLIER * atr

    size = gate.compute_size(
        equity=equity, entry_price=entry_price, stop_price=stop_price, calendar_multiplier=calendar_multiplier
    )

    assert size >= 0
    assert size == size  # not NaN
    assert size != float("inf")

    stop_distance = abs(entry_price - stop_price)
    dollar_risk = size * stop_distance
    max_risk = gate.RISK_PCT * equity * calendar_multiplier
    assert dollar_risk <= max_risk + 1e-6

    notional = size * entry_price
    assert notional <= gate.MAX_POSITION_PCT * equity + 1e-6


def test_size_is_zero_when_stop_equals_entry_atr_zero() -> None:
    """Degenerate input: ATR -> 0 means stop == entry -- must not divide by zero."""
    size = gate.compute_size(equity=10_000.0, entry_price=50.0, stop_price=50.0, calendar_multiplier=1.0)
    assert size == 0.0


def test_size_is_zero_when_equity_is_zero() -> None:
    """Degenerate input: equity -> 0."""
    size = gate.compute_size(equity=0.0, entry_price=50.0, stop_price=49.0, calendar_multiplier=1.0)
    assert size == 0.0


def test_size_handles_subpenny_price_without_error() -> None:
    """Degenerate input: price -> 0.01."""
    size = gate.compute_size(equity=10_000.0, entry_price=0.01, stop_price=0.005, calendar_multiplier=1.0)
    assert size >= 0
    assert size == size  # not NaN


def test_size_is_zero_when_calendar_multiplier_is_zero() -> None:
    """A STAND DOWN multiplier (0.0) must zero out risk entirely, not just reduce it."""
    size = gate.compute_size(equity=10_000.0, entry_price=100.0, stop_price=95.0, calendar_multiplier=0.0)
    assert size == 0.0


# --------------------------------------------------------------------------- #
# T011 -- Stops can never be widened (property test)
# --------------------------------------------------------------------------- #


@given(
    side=st.sampled_from(["long", "short"]),
    entry_price=st.floats(min_value=1, max_value=100_000, allow_nan=False, allow_infinity=False),
    old_stop_offset=st.floats(min_value=0.01, max_value=1_000, allow_nan=False, allow_infinity=False),
    new_stop_offset=st.floats(min_value=0.01, max_value=1_000, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=300)
def test_stop_change_is_never_allowed_to_widen(
    side: str, entry_price: float, old_stop_offset: float, new_stop_offset: float
) -> None:
    if side == "long":
        old_stop = entry_price - old_stop_offset
        new_stop = entry_price - new_stop_offset
    else:
        old_stop = entry_price + old_stop_offset
        new_stop = entry_price + new_stop_offset

    old_distance = abs(entry_price - old_stop)
    new_distance = abs(entry_price - new_stop)

    allowed = gate.validate_stop_change(side=side, entry_price=entry_price, old_stop=old_stop, new_stop=new_stop)

    if new_distance > old_distance + 1e-9:
        assert allowed is False, "widening a stop must never be allowed, regardless of caller"
    else:
        assert allowed is True


def test_stop_tightening_to_breakeven_is_allowed() -> None:
    assert gate.validate_stop_change(side="long", entry_price=100.0, old_stop=97.0, new_stop=100.0) is True


def test_exact_equal_distance_is_allowed_as_a_noop() -> None:
    assert gate.validate_stop_change(side="long", entry_price=100.0, old_stop=97.0, new_stop=97.0) is True


def test_tiny_widening_is_rejected() -> None:
    """Not just gross widening -- an epsilon-sized increase must also be caught."""
    assert gate.validate_stop_change(side="long", entry_price=100.0, old_stop=97.0, new_stop=96.99) is False


def test_short_side_widening_is_rejected() -> None:
    """The rule is side-aware: for a short, widening means the stop moving UP, away from entry."""
    assert gate.validate_stop_change(side="short", entry_price=100.0, old_stop=103.0, new_stop=103.01) is False


def test_short_side_tightening_is_allowed() -> None:
    assert gate.validate_stop_change(side="short", entry_price=100.0, old_stop=103.0, new_stop=101.5) is True


# --------------------------------------------------------------------------- #
# T012 -- Circuit breakers trip and hold (integration)
# --------------------------------------------------------------------------- #


def test_daily_loss_limit_halts_until_next_session() -> None:
    state = gate.BreakerState()
    now = datetime(2026, 9, 8, 15, 0, tzinfo=timezone.utc)  # a Tuesday

    state = gate.record_trade(state, r_multiple=gate.DAILY_LOSS_LIMIT_R, equity=9_800.0, now=now)

    assert gate.is_halted(state, now=now) is True
    next_session = datetime(2026, 9, 9, 14, 30, tzinfo=timezone.utc)
    assert gate.is_halted(state, now=next_session) is False


def test_daily_trade_cap_halts_until_next_session() -> None:
    state = gate.BreakerState()
    now = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)
    for i in range(gate.DAILY_TRADE_CAP):
        state = gate.record_trade(state, r_multiple=0.1, equity=10_000.0 + i, now=now + timedelta(minutes=i))

    assert gate.is_halted(state, now=now + timedelta(minutes=90)) is True
    next_session = now + timedelta(days=1)
    assert gate.is_halted(state, now=next_session) is False


def test_six_consecutive_losses_halts_for_24_hours() -> None:
    state = gate.BreakerState()
    now = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)
    equity = 10_000.0
    for i in range(gate.CONSECUTIVE_LOSS_HALT):
        equity -= 300.0
        state = gate.record_trade(state, r_multiple=-0.3, equity=equity, now=now + timedelta(minutes=i))

    assert gate.is_halted(state, now=now + timedelta(hours=1)) is True
    assert gate.is_halted(state, now=now + timedelta(hours=23)) is True
    assert gate.is_halted(state, now=now + timedelta(hours=25)) is False


def test_weekly_loss_limit_requires_manual_reset_past_the_week_boundary() -> None:
    """Unlike the daily limit, a weekly halt is not automatically lifted
    by the next session -- Risk_Policy.md Sec3: 'Manual, after a written
    review.'
    """
    state = gate.BreakerState()
    now = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)
    state = gate.record_trade(state, r_multiple=gate.WEEKLY_LOSS_LIMIT_R, equity=9_400.0, now=now)

    next_session = now + timedelta(days=1)
    assert gate.is_halted(state, now=next_session) is True
    far_future = now + timedelta(days=30)
    assert gate.is_halted(state, now=far_future) is True


def test_drawdown_kill_switch_never_auto_clears() -> None:
    state = gate.BreakerState(equity_peak=10_000.0)
    now = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)

    state = gate.record_trade(state, r_multiple=gate.DRAWDOWN_KILL_R, equity=8_500.0, now=now)

    assert state.hard_killed is True
    assert gate.is_halted(state, now=now) is True
    assert gate.is_halted(state, now=now + timedelta(days=365)) is True  # never auto-clears


def test_breaker_state_survives_a_process_restart(tmp_path: Path) -> None:
    """T012's own wording: 'Breakers survive a process restart (persisted,
    not in-memory).' `load_breaker_state` simulates a fresh process
    reading what a prior one wrote.
    """
    state = gate.BreakerState()
    now = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)
    state = gate.record_trade(state, r_multiple=gate.DAILY_LOSS_LIMIT_R, equity=9_800.0, now=now)

    path = tmp_path / "breakers.json"
    gate.save_breaker_state(state, path)
    reloaded = gate.load_breaker_state(path)

    assert gate.is_halted(reloaded, now=now) is True


# --------------------------------------------------------------------------- #
# T013 -- Pre-trade gate ordering (unit)
# --------------------------------------------------------------------------- #

_NOW = datetime(2026, 9, 8, 15, 0, tzinfo=timezone.utc)
_EQUITY = 10_000.0


def _passing_signal(**overrides: object) -> gate.Signal:
    base: dict[str, object] = dict(
        symbol="AAPL",
        side="long",
        entry_price=100.0,
        atr=1.0,  # stop = 100 - 1.5*1 = 98.5 -> 1.5% distance, inside the 0.5-5% sane band
        calendar_multiplier=1.0,
        rubric_score=10,
        rubric_r1=2,
        rubric_r6=2,
        quote_age_seconds=1.0,
        open_positions=0,
        sector_positions=0,
        broker_reconciled=True,
        card_written=True,
    )
    base.update(overrides)
    return gate.Signal(**base)  # type: ignore[arg-type]


def _passing_breaker_state(**overrides: object) -> gate.BreakerState:
    base: dict[str, object] = dict(
        session_r=0.0,
        trades_today=0,
        consecutive_losses=0,
        weekly_r=0.0,
        equity_peak=_EQUITY,
        halted_until=None,
        hard_killed=False,
    )
    base.update(overrides)
    return gate.BreakerState(**base)  # type: ignore[arg-type]


def test_a_fully_passing_signal_is_approved() -> None:
    decision = gate.evaluate(_passing_signal(), equity=_EQUITY, breaker_state=_passing_breaker_state(), now=_NOW)
    assert decision.status == "approved"
    assert decision.reject_reason is None
    assert decision.size > 0
    assert decision.stop_price is not None


def test_step1_kill_switch_rejects_before_anything_else() -> None:
    state = _passing_breaker_state(hard_killed=True)
    decision = gate.evaluate(_passing_signal(), equity=_EQUITY, breaker_state=state, now=_NOW)
    assert decision.reject_reason == "kill_switch"


def test_step2_daily_loss_limit_rejects() -> None:
    state = _passing_breaker_state(session_r=gate.DAILY_LOSS_LIMIT_R)
    decision = gate.evaluate(_passing_signal(), equity=_EQUITY, breaker_state=state, now=_NOW)
    assert decision.reject_reason == "daily_weekly_loss_limit"


def test_step3_trade_cap_rejects() -> None:
    state = _passing_breaker_state(trades_today=gate.DAILY_TRADE_CAP)
    decision = gate.evaluate(_passing_signal(), equity=_EQUITY, breaker_state=state, now=_NOW)
    assert decision.reject_reason == "trade_cap"


def test_step4_calendar_stand_down_rejects() -> None:
    signal = _passing_signal(calendar_multiplier=0.0)
    decision = gate.evaluate(signal, equity=_EQUITY, breaker_state=_passing_breaker_state(), now=_NOW)
    assert decision.reject_reason == "calendar_stand_down"


def test_step5_max_concurrent_positions_rejects() -> None:
    signal = _passing_signal(open_positions=gate.MAX_CONCURRENT_POSITIONS)
    decision = gate.evaluate(signal, equity=_EQUITY, breaker_state=_passing_breaker_state(), now=_NOW)
    assert decision.reject_reason == "max_concurrent_positions"


def test_step6_sector_cap_rejects() -> None:
    signal = _passing_signal(sector_positions=gate.MAX_SECTOR_POSITIONS)
    decision = gate.evaluate(signal, equity=_EQUITY, breaker_state=_passing_breaker_state(), now=_NOW)
    assert decision.reject_reason == "sector_correlation_cap"


def test_step7_stale_quote_rejects() -> None:
    signal = _passing_signal(quote_age_seconds=gate.MAX_QUOTE_AGE_SECONDS + 0.01)
    decision = gate.evaluate(signal, equity=_EQUITY, breaker_state=_passing_breaker_state(), now=_NOW)
    assert decision.reject_reason == "quote_freshness"


def test_step8_broker_state_drift_halts_not_rejects() -> None:
    """Step 8 is qualitatively different: a HALT, not a REJECT."""
    signal = _passing_signal(broker_reconciled=False)
    decision = gate.evaluate(signal, equity=_EQUITY, breaker_state=_passing_breaker_state(), now=_NOW)
    assert decision.status == "halted"
    assert decision.reject_reason == "broker_reconciliation"


def test_step9_zero_size_rejects() -> None:
    """ATR -> 0 collapses the stop onto entry, so computed size is 0 --
    caught explicitly at step 9, distinct from the calendar-multiplier
    path (step 4) which also zeroes size but must be caught earlier.
    """
    signal = _passing_signal(atr=0.0)
    decision = gate.evaluate(signal, equity=_EQUITY, breaker_state=_passing_breaker_state(), now=_NOW)
    assert decision.reject_reason == "size_within_caps"


def test_step10_stop_distance_too_tight_rejects() -> None:
    signal = _passing_signal(atr=0.0001)  # stop distance << 0.5% of entry
    decision = gate.evaluate(signal, equity=_EQUITY, breaker_state=_passing_breaker_state(), now=_NOW)
    assert decision.reject_reason == "stop_distance_sane"


def test_step10_stop_distance_too_wide_rejects() -> None:
    signal = _passing_signal(atr=50.0)  # stop distance = 1.5*50 = 75, i.e. 75% of entry
    decision = gate.evaluate(signal, equity=_EQUITY, breaker_state=_passing_breaker_state(), now=_NOW)
    assert decision.reject_reason == "stop_distance_sane"


def test_step11_rubric_score_too_low_rejects() -> None:
    signal = _passing_signal(rubric_score=8)
    decision = gate.evaluate(signal, equity=_EQUITY, breaker_state=_passing_breaker_state(), now=_NOW)
    assert decision.reject_reason == "rubric_threshold"


def test_step11_rubric_r1_zero_rejects_even_with_high_total_score() -> None:
    """The hard per-component requirement (R1 >= 1) can't be bought back
    by a high total score -- Strategy_Spec.md's rubric gate, enforced here.
    """
    signal = _passing_signal(rubric_score=11, rubric_r1=0, rubric_r6=2)
    decision = gate.evaluate(signal, equity=_EQUITY, breaker_state=_passing_breaker_state(), now=_NOW)
    assert decision.reject_reason == "rubric_threshold"


def test_step11_rubric_r6_zero_rejects_even_with_high_total_score() -> None:
    signal = _passing_signal(rubric_score=11, rubric_r1=2, rubric_r6=0)
    decision = gate.evaluate(signal, equity=_EQUITY, breaker_state=_passing_breaker_state(), now=_NOW)
    assert decision.reject_reason == "rubric_threshold"


def test_step12_missing_trade_card_rejects_even_when_everything_else_passes() -> None:
    signal = _passing_signal(card_written=False)
    decision = gate.evaluate(signal, equity=_EQUITY, breaker_state=_passing_breaker_state(), now=_NOW)
    assert decision.reject_reason == "trade_card_written"


def test_earlier_failure_short_circuits_a_later_one() -> None:
    """Signal fails BOTH step 2 (daily loss limit) and step 11 (rubric) --
    the reported reason must be step 2's, proving steps evaluate in order
    and the first failure wins, not the "most severe" or "last checked."
    """
    state = _passing_breaker_state(session_r=gate.DAILY_LOSS_LIMIT_R)
    signal = _passing_signal(rubric_score=2, rubric_r1=0, rubric_r6=0)
    decision = gate.evaluate(signal, equity=_EQUITY, breaker_state=state, now=_NOW)
    assert decision.reject_reason == "daily_weekly_loss_limit"


# --------------------------------------------------------------------------- #
# T014 -- Calendar STAND DOWN blocks everything (M004+M006 integration)
# --------------------------------------------------------------------------- #


def test_calendar_stand_down_blocks_even_a_perfect_signal() -> None:
    """Multiplier 0.00 must reject regardless of rubric score -- a
    12-out-of-12 rubric candidate is blocked exactly the same as a
    2-out-of-12 one once the calendar gate says STAND DOWN.
    """
    signal = _passing_signal(calendar_multiplier=0.0, rubric_score=12, rubric_r1=2, rubric_r6=2)
    decision = gate.evaluate(signal, equity=_EQUITY, breaker_state=_passing_breaker_state(), now=_NOW)

    assert decision.status == "rejected"
    assert decision.reject_reason == "calendar_stand_down"
    assert decision.size == 0.0
