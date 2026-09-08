"""Tests for M005 -- vt.signal.rubric (T007; see 06_Tests.md).

Phase D (`16_Next_Steps.md`): T007 -- rubric scoring and thresholds.
Score sums correctly 0-12; candidate with total 10 but R1=0 is
rejected (hard requirement); breakdown emitted for all six components.

Golden values are derived by hand from Strategy_Spec.md section 3, one
component boundary at a time -- the point is to catch a spec-mismatch,
not to snapshot arbitrary numbers.

Run with: pytest vt/tests -m unit
"""

from __future__ import annotations

import pytest

from vt.signal import rubric

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fixture builders -- one perfect candidate, plus overrides per test
# --------------------------------------------------------------------------- #


def _perfect(**overrides) -> rubric.Candidate:
    """A candidate that scores 2 on every component -- the reference
    point every 'break one component' test starts from."""
    defaults = dict(
        symbol="AAPL",
        venue="alpaca",
        price=180.0,
        vwap=175.0,
        ema9=178.0,
        ema21=176.0,
        ema9_rising=True,
        ema21_rising=True,
        rsi14=62.0,  # in (55, 70]
        rvol=4.0,  # > 3
        obv_slope=1.5,  # rising
        adx14=30.0,  # > 25
        atr_expanding=True,
        prior_close=170.0,
        opening_range_high=178.0,
        opening_range_low=172.0,
        broke_opening_range_high=True,
        held_on_retest=True,
        relative_strength_pct=1.5,  # > 1%
    )
    defaults.update(overrides)
    return rubric.Candidate(**defaults)


# --------------------------------------------------------------------------- #
# Total range and shape
# --------------------------------------------------------------------------- #


def test_perfect_candidate_scores_twelve_and_is_eligible() -> None:
    s = rubric.score(_perfect())

    assert s.total == 12
    assert s.breakdown == {"R1": 2, "R2": 2, "R3": 2, "R4": 2, "R5": 2, "R6": 2}
    assert s.entry_eligible is True
    assert s.ineligible_reason is None
    # Full breakdown emitted for the Trade Card / component-attribution.
    assert set(s.breakdown.keys()) == {"R1", "R2", "R3", "R4", "R5", "R6"}
    assert len(s.reasons) == 6


def test_worst_candidate_scores_zero_and_is_ineligible() -> None:
    # All components fail: price below VWAP, RSI too low, RVOL too low,
    # ADX flat, price below prior close, underperforming benchmark.
    c = _perfect(
        price=170.0,
        vwap=175.0,
        rsi14=30.0,
        rvol=0.5,
        obv_slope=-1.0,
        adx14=10.0,
        atr_expanding=False,
        prior_close=175.0,
        broke_opening_range_high=False,
        held_on_retest=False,
        relative_strength_pct=-2.0,
    )
    s = rubric.score(c)
    assert s.total == 0
    assert s.breakdown == {"R1": 0, "R2": 0, "R3": 0, "R4": 0, "R5": 0, "R6": 0}
    assert s.entry_eligible is False
    assert s.ineligible_reason == "total_below_threshold"


def test_breakdown_values_are_always_in_the_zero_to_two_range() -> None:
    s = rubric.score(_perfect())
    for key, pts in s.breakdown.items():
        assert pts in (0, 1, 2), f"{key} scored {pts}, outside 0-2"


# --------------------------------------------------------------------------- #
# T007 headline case -- total 10 but R1=0 is REJECTED (hard requirement)
# --------------------------------------------------------------------------- #


def test_total_ten_with_r1_zero_is_ineligible() -> None:
    """The single most important assertion T007 was written for.
    Strategy_Spec.md section 3: R1 must be >= 1 regardless of total.
    Set up a candidate with R1=0 and every other component at 2 (total
    10). Must be rejected."""
    c = _perfect(price=170.0, vwap=175.0)  # R1=0 (price below VWAP)
    s = rubric.score(c)

    assert s.breakdown["R1"] == 0
    assert s.total == 10  # 0 + 2 + 2 + 2 + 2 + 2
    assert s.total >= rubric.ENTRY_SCORE_THRESHOLD  # would pass on total alone
    assert s.entry_eligible is False
    assert s.ineligible_reason == "r1_below_floor"


def test_total_ten_with_r6_zero_is_ineligible() -> None:
    """The other hard floor: no laggards. R6 must be >= 1 regardless of
    total."""
    c = _perfect(relative_strength_pct=-2.0)  # R6=0
    s = rubric.score(c)

    assert s.breakdown["R6"] == 0
    assert s.total == 10
    assert s.entry_eligible is False
    assert s.ineligible_reason == "r6_below_floor"


def test_score_at_threshold_with_hard_floors_met_is_eligible() -> None:
    """Score == 9 exactly, R1=1, R6=1 -> the boundary case where the
    gate should let the signal through. Off-by-one bugs here would
    quietly leak (or block) real signals."""
    # R1=1 (above VWAP, EMAs unaligned), R2=2, R3=2, R4=2, R5=1, R6=1
    # total = 9
    c = _perfect(
        ema9=176.0,
        ema21=178.0,  # EMAs unaligned -> R1=1
        broke_opening_range_high=False,
        held_on_retest=False,  # R5 falls back to 1 (above prior close)
        relative_strength_pct=0.5,  # R6=1 (in line)
    )
    s = rubric.score(c)
    assert s.breakdown["R1"] == 1
    assert s.breakdown["R5"] == 1
    assert s.breakdown["R6"] == 1
    assert s.total == 9
    assert s.entry_eligible is True
    assert s.ineligible_reason is None


def test_score_one_below_threshold_is_ineligible_even_with_hard_floors_met() -> None:
    """Total=8, R1>=1, R6>=1 -> not eligible. The total threshold is
    the third check, not the first, but it still gates."""
    # R1=1, R2=2, R3=2, R4=1 (ADX in mid), R5=1, R6=1 -> total 8
    c = _perfect(
        ema9=176.0,
        ema21=178.0,  # R1=1
        adx14=20.0,
        atr_expanding=False,  # R4=1
        broke_opening_range_high=False,
        held_on_retest=False,  # R5=1
        relative_strength_pct=0.5,  # R6=1
    )
    s = rubric.score(c)
    assert s.total == 8
    assert s.entry_eligible is False
    assert s.ineligible_reason == "total_below_threshold"


# --------------------------------------------------------------------------- #
# R1 -- trend alignment
# --------------------------------------------------------------------------- #


def test_r1_zero_when_price_below_vwap() -> None:
    assert rubric.score(_perfect(price=170.0, vwap=175.0)).breakdown["R1"] == 0


def test_r1_zero_when_price_equals_vwap() -> None:
    # Strictly above VWAP is required; equal is not enough.
    assert rubric.score(_perfect(price=175.0, vwap=175.0)).breakdown["R1"] == 0


def test_r1_one_when_above_vwap_but_emas_unaligned() -> None:
    c = _perfect(ema9=176.0, ema21=178.0)  # EMA9 < EMA21
    assert rubric.score(c).breakdown["R1"] == 1


def test_r1_one_when_above_vwap_and_aligned_but_not_both_rising() -> None:
    c = _perfect(ema21_rising=False)
    assert rubric.score(c).breakdown["R1"] == 1


# --------------------------------------------------------------------------- #
# R2 -- RSI momentum
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "rsi,expected",
    [
        (30.0, 0),
        (44.9, 0),
        (45.0, 1),
        (50.0, 1),
        (55.0, 1),
        (55.01, 2),
        (62.0, 2),
        (70.0, 2),
        (70.01, 0),  # 70-80 gap treated as 0 (documented in rubric.py)
        (85.0, 0),
    ],
)
def test_r2_rsi_zones(rsi: float, expected: int) -> None:
    assert rubric.score(_perfect(rsi14=rsi)).breakdown["R2"] == expected


# --------------------------------------------------------------------------- #
# R3 -- RVOL + OBV
# --------------------------------------------------------------------------- #


def test_r3_zero_when_rvol_below_two() -> None:
    assert rubric.score(_perfect(rvol=1.9)).breakdown["R3"] == 0


def test_r3_one_when_rvol_two_to_three_and_obv_flat() -> None:
    c = _perfect(rvol=2.5, obv_slope=0.0)
    assert rubric.score(c).breakdown["R3"] == 1


def test_r3_two_requires_both_rvol_above_three_and_obv_rising() -> None:
    assert rubric.score(_perfect(rvol=4.0, obv_slope=1.5)).breakdown["R3"] == 2
    # RVOL high but OBV falling -> falls back to 0 (volume without
    # accumulation is distribution, not confirmation).
    assert rubric.score(_perfect(rvol=4.0, obv_slope=-1.0)).breakdown["R3"] == 0
    # RVOL high but OBV flat -> demoted to 1, not 2.
    assert rubric.score(_perfect(rvol=4.0, obv_slope=0.0)).breakdown["R3"] == 1


# --------------------------------------------------------------------------- #
# R4 -- ADX + ATR expansion
# --------------------------------------------------------------------------- #


def test_r4_zero_when_adx_below_fifteen() -> None:
    assert rubric.score(_perfect(adx14=10.0)).breakdown["R4"] == 0


def test_r4_one_when_adx_in_the_fifteen_to_twentyfive_band() -> None:
    assert rubric.score(_perfect(adx14=20.0, atr_expanding=False)).breakdown["R4"] == 1


def test_r4_two_requires_adx_above_twentyfive_and_atr_expanding() -> None:
    assert rubric.score(_perfect(adx14=30.0, atr_expanding=True)).breakdown["R4"] == 2
    # ADX high but ATR not expanding -> demoted to 1.
    assert rubric.score(_perfect(adx14=30.0, atr_expanding=False)).breakdown["R4"] == 1


# --------------------------------------------------------------------------- #
# R5 -- structure
# --------------------------------------------------------------------------- #


def test_r5_zero_below_prior_close() -> None:
    assert rubric.score(_perfect(price=169.0, prior_close=170.0)).breakdown["R5"] == 0


def test_r5_two_when_broke_or_high_and_held_on_retest() -> None:
    c = _perfect(broke_opening_range_high=True, held_on_retest=True)
    assert rubric.score(c).breakdown["R5"] == 2


def test_r5_one_when_broke_but_did_not_hold_on_retest() -> None:
    c = _perfect(broke_opening_range_high=True, held_on_retest=False)
    assert rubric.score(c).breakdown["R5"] == 1


def test_r5_one_when_inside_the_opening_range() -> None:
    c = _perfect(
        price=175.0,
        opening_range_low=172.0,
        opening_range_high=178.0,
        broke_opening_range_high=False,
        held_on_retest=False,
    )
    assert rubric.score(c).breakdown["R5"] == 1


# --------------------------------------------------------------------------- #
# R6 -- relative strength
# --------------------------------------------------------------------------- #


def test_r6_zero_when_underperforming() -> None:
    assert rubric.score(_perfect(relative_strength_pct=-0.5)).breakdown["R6"] == 0


def test_r6_one_when_in_line() -> None:
    assert rubric.score(_perfect(relative_strength_pct=0.5)).breakdown["R6"] == 1
    assert rubric.score(_perfect(relative_strength_pct=1.0)).breakdown["R6"] == 1


def test_r6_two_when_outperforming_by_more_than_one_pct() -> None:
    assert rubric.score(_perfect(relative_strength_pct=1.5)).breakdown["R6"] == 2


# --------------------------------------------------------------------------- #
# Rank
# --------------------------------------------------------------------------- #


def test_rank_sorts_eligible_first_then_by_total_desc_then_by_symbol() -> None:
    good_high = _perfect(symbol="AAPL")  # 12, eligible
    good_low = _perfect(symbol="ZZZZ")  # 12, eligible -- but sorts after AAPL on tie
    good_mid = _perfect(
        symbol="MSFT",
        ema9=176.0,
        ema21=178.0,  # R1=1 -> total 11
    )
    ineligible_high_total = _perfect(
        symbol="BBBB", price=170.0, vwap=175.0
    )  # total 10, but R1=0 -> ineligible

    ranked = rubric.rank([ineligible_high_total, good_low, good_mid, good_high])

    # Eligible first, sorted by total desc then symbol asc.
    assert [s.symbol for s in ranked] == ["AAPL", "ZZZZ", "MSFT", "BBBB"]
    assert [s.entry_eligible for s in ranked] == [True, True, True, False]


def test_rank_returns_empty_list_for_empty_input() -> None:
    assert rubric.rank([]) == []


# --------------------------------------------------------------------------- #
# Reasons stay in R1..R6 order so the Trade Card renders deterministically
# --------------------------------------------------------------------------- #


def test_reasons_ordered_r1_through_r6() -> None:
    s = rubric.score(_perfect())
    assert [r.split("=")[0] for r in s.reasons] == ["R1", "R2", "R3", "R4", "R5", "R6"]
