"""Tests for M004 -- vt.gate.calendar (T006; see 06_Tests.md).

Phase C3 (`16_Next_Steps.md`): written against a STUB where every
`vt.gate.calendar` function raises NotImplementedError. Every test in
this file must fail with NotImplementedError before the real
implementation lands. Spec source: `Strategy_Spec.md` section 2.

T014 (calendar STAND DOWN blocks everything at M006) is covered in
test_gate.py already, using an injected `calendar_multiplier` on the
test Signal -- M004 does not need to be wired into M006 for that test.
This file exercises M004's own state machine in isolation.

Run with: pytest vt/tests -m unit
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from vt.gate import calendar as cal

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def events_yaml(tmp_path: Path) -> Path:
    """Isolated events YAML so tests don't depend on the bundled seed."""
    path = tmp_path / "events.yaml"
    path.write_text(
        "stand_down:\n"
        "  - date: 2026-09-17\n"
        "    event: FOMC\n"
        "  - date: 2026-09-11\n"
        "    event: CPI\n"
        "  - date: 2026-09-05\n"
        "    event: NFP\n",
        encoding="utf-8",
    )
    return path


def _quiet_history(n: int = 60, base: float = 0.010) -> list[float]:
    """Trailing realized-vol history where `base` sits well above the 20th pct."""
    return [base + 0.0001 * i for i in range(n)]


# --------------------------------------------------------------------------- #
# T006 -- the four canonical scenarios from Strategy_Spec section 2
# --------------------------------------------------------------------------- #


def test_t006_fomc_morning_stand_down(events_yaml: Path) -> None:
    """FOMC release day, pre-print -> STAND_DOWN (0.00)."""
    result = cal.gate_state(
        datetime(2026, 9, 17, 9, 30, tzinfo=timezone.utc),
        vix=15.0,
        realized_vol=0.015,
        realized_vol_history=_quiet_history(),
        index_returns=[0.001, -0.002, 0.0005, -0.0007, 0.0003],
        events_path=events_yaml,
    )
    assert result.state == "STAND_DOWN"
    assert result.multiplier == cal.MULT_STAND_DOWN == 0.00
    assert any("FOMC" in r for r in result.reasons)


def test_t006_vix_thirty_reduced(events_yaml: Path) -> None:
    """VIX 30 (>28) on an otherwise quiet Tuesday -> REDUCED (0.50)."""
    result = cal.gate_state(
        datetime(2026, 9, 15, 14, 0, tzinfo=timezone.utc),  # Tuesday, no event
        vix=30.0,
        realized_vol=0.015,
        realized_vol_history=_quiet_history(),
        index_returns=[0.001, -0.002, 0.0005, -0.0007, 0.0003],
        events_path=events_yaml,
    )
    assert result.state == "REDUCED"
    assert result.multiplier == cal.MULT_REDUCED == 0.50
    assert any("vix" in r.lower() for r in result.reasons)


def test_t006_quiet_tuesday_normal(events_yaml: Path) -> None:
    """No triggers -> NORMAL (1.00)."""
    result = cal.gate_state(
        datetime(2026, 9, 15, 14, 0, tzinfo=timezone.utc),  # Tuesday, no event
        vix=15.0,
        realized_vol=0.015,
        realized_vol_history=_quiet_history(),
        index_returns=[0.001, -0.002, 0.0005, -0.0007, 0.0003],
        events_path=events_yaml,
    )
    assert result.state == "NORMAL"
    assert result.multiplier == cal.MULT_NORMAL == 1.00


def test_t006_realized_vol_below_p20_stand_down(events_yaml: Path) -> None:
    """Today's realized vol below 20th pct of trailing 60d -> STAND_DOWN."""
    history = _quiet_history(n=60, base=0.020)  # p20 well above 0.005
    result = cal.gate_state(
        datetime(2026, 9, 15, 14, 0, tzinfo=timezone.utc),
        vix=15.0,
        realized_vol=0.005,
        realized_vol_history=history,
        index_returns=[0.001, -0.002, 0.0005, -0.0007, 0.0003],
        events_path=events_yaml,
    )
    assert result.state == "STAND_DOWN"
    assert result.multiplier == 0.00
    assert any("realized_vol" in r for r in result.reasons)


# --------------------------------------------------------------------------- #
# Precedence & invariants
# --------------------------------------------------------------------------- #


def test_stand_down_wins_over_reduced(events_yaml: Path) -> None:
    """FOMC day AND VIX>28 -> STAND_DOWN, but the VIX reason is still logged."""
    result = cal.gate_state(
        datetime(2026, 9, 17, 9, 30, tzinfo=timezone.utc),
        vix=35.0,
        realized_vol=0.015,
        realized_vol_history=_quiet_history(),
        events_path=events_yaml,
    )
    assert result.state == "STAND_DOWN"
    assert result.multiplier == 0.00
    assert any("FOMC" in r for r in result.reasons)
    assert any("vix" in r.lower() for r in result.reasons)


def test_multiplier_is_always_one_of_three(events_yaml: Path) -> None:
    """The multiplier lives in {0.00, 0.50, 1.00} across many inputs."""
    scenarios = [
        # (asof, vix, realized_vol)
        (datetime(2026, 9, 15, 14, 0, tzinfo=timezone.utc), None, None),
        (datetime(2026, 9, 15, 14, 0, tzinfo=timezone.utc), 10.0, 0.02),
        (datetime(2026, 9, 15, 14, 0, tzinfo=timezone.utc), 45.0, 0.02),
        (datetime(2026, 9, 17, 9, 30, tzinfo=timezone.utc), 20.0, 0.02),
        (datetime(2026, 9, 18, 14, 0, tzinfo=timezone.utc), 20.0, 0.02),  # quad witching
    ]
    for asof, vix, rv in scenarios:
        result = cal.gate_state(
            asof,
            vix=vix,
            realized_vol=rv,
            realized_vol_history=_quiet_history(),
            events_path=events_yaml,
        )
        assert result.multiplier in {0.00, 0.50, 1.00}, (asof, vix, rv, result)


def test_normal_state_has_empty_reasons(events_yaml: Path) -> None:
    result = cal.gate_state(
        datetime(2026, 9, 15, 14, 0, tzinfo=timezone.utc),
        vix=15.0,
        realized_vol=0.015,
        realized_vol_history=_quiet_history(),
        events_path=events_yaml,
    )
    assert result.state == "NORMAL"
    assert result.reasons == ()


def test_non_normal_states_always_carry_reasons(events_yaml: Path) -> None:
    stand_down = cal.gate_state(
        datetime(2026, 9, 17, 9, 30, tzinfo=timezone.utc),
        events_path=events_yaml,
    )
    assert stand_down.state == "STAND_DOWN"
    assert len(stand_down.reasons) >= 1

    reduced = cal.gate_state(
        datetime(2026, 9, 15, 14, 0, tzinfo=timezone.utc),
        vix=35.0,
        events_path=events_yaml,
    )
    assert reduced.state == "REDUCED"
    assert len(reduced.reasons) >= 1


# --------------------------------------------------------------------------- #
# Rule-specific behaviour
# --------------------------------------------------------------------------- #


def test_prior_session_two_sigma_move_triggers_reduced(events_yaml: Path) -> None:
    """Last index return > 2 * stdev(prior returns) -> REDUCED."""
    prior = [0.001, -0.002, 0.0015, -0.001, 0.0005, -0.0008, 0.0012, -0.0011]
    latest = 0.05  # far beyond 2 sigma of `prior`
    result = cal.gate_state(
        datetime(2026, 9, 15, 14, 0, tzinfo=timezone.utc),
        vix=15.0,
        realized_vol=0.015,
        realized_vol_history=_quiet_history(),
        index_returns=[*prior, latest],
        events_path=events_yaml,
    )
    assert result.state == "REDUCED"
    assert any("sigma" in r for r in result.reasons)


def test_sub_two_sigma_move_stays_normal(events_yaml: Path) -> None:
    prior = [0.001, -0.002, 0.0015, -0.001, 0.0005, -0.0008, 0.0012, -0.0011]
    latest = 0.001
    result = cal.gate_state(
        datetime(2026, 9, 15, 14, 0, tzinfo=timezone.utc),
        vix=15.0,
        realized_vol=0.015,
        realized_vol_history=_quiet_history(),
        index_returns=[*prior, latest],
        events_path=events_yaml,
    )
    assert result.state == "NORMAL"


def test_quad_witching_third_friday_of_quarter_end_triggers_reduced(
    events_yaml: Path,
) -> None:
    # 2026-09-18 is the third Friday of September -> quad witching.
    result = cal.gate_state(
        datetime(2026, 9, 18, 14, 0, tzinfo=timezone.utc),
        vix=15.0,
        realized_vol=0.015,
        realized_vol_history=_quiet_history(),
        events_path=events_yaml,
    )
    assert result.state == "REDUCED"
    assert any("quad_witching" in r for r in result.reasons)


def test_monthly_opex_third_friday_of_non_quarter_end_triggers_reduced(
    events_yaml: Path,
) -> None:
    # 2026-10-16 is the third Friday of October -> monthly OPEX (not quad).
    result = cal.gate_state(
        datetime(2026, 10, 16, 14, 0, tzinfo=timezone.utc),
        vix=15.0,
        realized_vol=0.015,
        realized_vol_history=_quiet_history(),
        events_path=events_yaml,
    )
    assert result.state == "REDUCED"
    assert any("monthly_opex" in r for r in result.reasons)


def test_second_friday_is_not_opex(events_yaml: Path) -> None:
    # 2026-09-11 IS an event (CPI) so we'd get STAND_DOWN; use a non-event 2nd Fri.
    # 2026-10-09 is the second Friday of October -> NOT OPEX.
    result = cal.gate_state(
        datetime(2026, 10, 9, 14, 0, tzinfo=timezone.utc),
        vix=15.0,
        realized_vol=0.015,
        realized_vol_history=_quiet_history(),
        events_path=events_yaml,
    )
    assert result.state == "NORMAL"


# --------------------------------------------------------------------------- #
# Fail-safe: missing inputs must not spuriously downgrade the state
# --------------------------------------------------------------------------- #


def test_missing_vix_does_not_trigger_reduced(events_yaml: Path) -> None:
    result = cal.gate_state(
        datetime(2026, 9, 15, 14, 0, tzinfo=timezone.utc),
        vix=None,
        realized_vol=0.015,
        realized_vol_history=_quiet_history(),
        events_path=events_yaml,
    )
    assert result.state == "NORMAL"


def test_missing_realized_vol_does_not_trigger_stand_down(events_yaml: Path) -> None:
    result = cal.gate_state(
        datetime(2026, 9, 15, 14, 0, tzinfo=timezone.utc),
        vix=15.0,
        realized_vol=None,
        realized_vol_history=None,
        events_path=events_yaml,
    )
    assert result.state == "NORMAL"


def test_missing_index_returns_does_not_trigger_reduced(events_yaml: Path) -> None:
    result = cal.gate_state(
        datetime(2026, 9, 15, 14, 0, tzinfo=timezone.utc),
        vix=15.0,
        realized_vol=0.015,
        realized_vol_history=_quiet_history(),
        index_returns=None,
        events_path=events_yaml,
    )
    assert result.state == "NORMAL"


# --------------------------------------------------------------------------- #
# YAML loading
# --------------------------------------------------------------------------- #


def test_missing_events_yaml_is_not_fatal(tmp_path: Path) -> None:
    """A missing events file is treated as 'no macro prints', not a crash."""
    result = cal.gate_state(
        datetime(2026, 9, 17, 9, 30, tzinfo=timezone.utc),
        vix=15.0,
        realized_vol=0.015,
        realized_vol_history=_quiet_history(),
        events_path=tmp_path / "nope.yaml",
    )
    assert result.state == "NORMAL"


def test_bundled_events_yaml_covers_known_seed_dates() -> None:
    """The bundled seed YAML flags known FOMC/CPI/NFP dates in its window."""
    result = cal.gate_state(
        datetime(2026, 9, 17, 9, 30, tzinfo=timezone.utc),
        vix=15.0,
        realized_vol=0.015,
        realized_vol_history=_quiet_history(),
    )
    assert result.state == "STAND_DOWN"


# --------------------------------------------------------------------------- #
# Percentile helper -- surface-level correctness
# --------------------------------------------------------------------------- #


def test_percentile_linear_interpolation() -> None:
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert cal._percentile(xs, 0) == 1.0
    assert cal._percentile(xs, 50) == 3.0
    assert cal._percentile(xs, 100) == 5.0
    # 20th pct of a 5-elt list: k = 0.20 * 4 = 0.8 -> 1 + 0.8*(2-1) = 1.8
    assert cal._percentile(xs, 20) == pytest.approx(1.8)


def test_quad_witching_and_opex_helpers_sanity() -> None:
    assert cal._is_quad_witching(date(2026, 9, 18)) is True   # 3rd Fri of Sep
    assert cal._is_quad_witching(date(2026, 10, 16)) is False  # Oct not a quad month
    assert cal._is_monthly_opex(date(2026, 10, 16)) is True    # 3rd Fri of Oct
    assert cal._is_monthly_opex(date(2026, 10, 9)) is False    # 2nd Fri
    assert cal._is_monthly_opex(date(2026, 10, 15)) is False   # Thursday
