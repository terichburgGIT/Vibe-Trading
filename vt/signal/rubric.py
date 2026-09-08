"""M005 -- Signal Scorer. Phase D, T007.

Applies the 6-component rubric from `Strategy_Spec.md` section 3,
produces a 0-12 score plus per-component breakdown, and ranks
candidates. Net-new. Equal weights, locked until M010 exists (AD007) --
weight-fitting on the same data used to pick the components is the
textbook overfit.

This is the module that actually generates the daily trade hypothesis --
deterministically, from the rubric -- not the LLM (`vt.analyst`, M011,
narrates an already-scored candidate; AD005). The per-component
breakdown this module emits is what the Trade Card renders as "the
reasons for the pick," and what post-hoc component-attribution analysis
uses to figure out which rubric component was carrying the edge (which
is most of the value of keeping the breakdown at all -- a bare 0-12
number tells you a signal fired, not why, and "not why" makes the
system undebuggable when it stops working).

Full contract in `03_Modules.md` section M005.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Rubric thresholds -- Strategy_Spec.md section 3, in one place so the
# tests can import and refer to them by name (which is what makes T007
# a spec check, not just a golden-value check).

# R2 -- RSI14 on 5-min
RSI_ZONE_1_MIN = 45.0  # 45-55 inclusive -> 1 point
RSI_ZONE_1_MAX = 55.0
RSI_ZONE_2_MAX = 70.0  # 55 < RSI <= 70 -> 2 points ("strong but not exhausted")
RSI_OVERBOUGHT = 80.0  # spec says "> 80" is 0; 70-80 gap is treated as 0
# too (conservative -- 70+ RSI is not what the rubric rewards).

# R3 -- RVOL
RVOL_ZONE_1_MIN = 2.0  # 2-3 with flat OBV -> 1 point
RVOL_ZONE_2_MIN = 3.0  # > 3 with rising OBV -> 2 points

# R4 -- ADX14
ADX_ZONE_1_MIN = 15.0  # 15-25 -> 1 point
ADX_ZONE_2_MIN = 25.0  # > 25 with ATR expanding -> 2 points

# R6 -- relative strength vs benchmark
RELATIVE_STRENGTH_OUTPERFORM_PCT = 1.0  # > 1% outperformance -> 2 points

ENTRY_SCORE_THRESHOLD = 9  # Strategy_Spec.md section 3
HARD_MIN_R1 = 1  # R1 >= 1 is a hard requirement regardless of total
HARD_MIN_R6 = 1  # R6 >= 1 is a hard requirement regardless of total

COMPONENT_KEYS: tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5", "R6")


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Candidate:
    """The pre-computed inputs one call of `score()` needs. Fed by
    upstream modules -- M002 supplies RVOL, M003 supplies VWAP / EMA /
    RSI / OBV slope / ADX / ATR-expansion, the caller supplies the R5
    structural facts and the R6 relative-strength number.

    Kept explicit rather than accepting a raw price series so this
    module is deterministic and cheap to unit-test -- the rubric has one
    job (turn 6 inputs into a 0-12 score) and mixing indicator math into
    it would blur where each component's edge actually comes from.
    """

    symbol: str
    venue: str

    # R1 inputs
    price: float
    vwap: float
    ema9: float
    ema21: float
    ema9_rising: bool
    ema21_rising: bool

    # R2 input
    rsi14: float

    # R3 inputs
    rvol: float
    obv_slope: float  # positive -> rising, ~0 -> flat, negative -> falling

    # R4 inputs
    adx14: float
    atr_expanding: bool

    # R5 inputs -- pre-computed structural facts
    prior_close: float
    opening_range_high: float
    opening_range_low: float
    broke_opening_range_high: bool
    held_on_retest: bool

    # R6 input -- outperformance vs benchmark (SPY / BTC) over trailing
    # 30 min, in percent. Positive = outperforming.
    relative_strength_pct: float


@dataclass(frozen=True)
class Score:
    """Full rubric output for one candidate. `breakdown` is fixed to the
    six R1-R6 keys so callers can iterate or JSON-serialize it without
    caring about which fields exist. `reasons` is the human-readable
    render of the same info -- what the Trade Card shows as chips.

    `entry_eligible` is the single source of truth for "should this
    signal be forwarded to M006's gate": total >= ENTRY_SCORE_THRESHOLD
    AND R1 >= HARD_MIN_R1 AND R6 >= HARD_MIN_R6. A total-10 signal with
    R1=0 is NOT eligible, and that's the point of the hard floors
    (Strategy_Spec.md section 3 -- 'no counter-trend entries, no
    laggards, regardless of total score').
    """

    symbol: str
    venue: str
    total: int
    breakdown: dict[str, int]  # R1..R6 -> 0/1/2
    reasons: tuple[str, ...]  # one per component, in R1..R6 order
    entry_eligible: bool
    ineligible_reason: str | None  # 'total_below_threshold' / 'r1_below_floor' / 'r6_below_floor' / None


# --------------------------------------------------------------------------- #
# Per-component scorers
# --------------------------------------------------------------------------- #


def _score_r1(c: Candidate) -> tuple[int, str]:
    """Trend alignment -- price vs VWAP and EMA(9)/EMA(21).
    0: price on wrong side of VWAP.
    1: price above VWAP, EMAs unaligned or not both rising.
    2: price above VWAP AND EMA9 > EMA21 AND both rising.
    """
    if c.price <= c.vwap:
        return 0, f"R1=0 price {c.price:.2f} <= VWAP {c.vwap:.2f}"
    if c.ema9 > c.ema21 and c.ema9_rising and c.ema21_rising:
        return 2, f"R1=2 above VWAP, EMA9>EMA21, both rising"
    return 1, f"R1=1 above VWAP, EMAs unaligned"


def _score_r2(c: Candidate) -> tuple[int, str]:
    """Momentum -- RSI14 on 5-min.
    0: RSI < 45 or > 70 (the 70-80 gap in the spec is treated as 0 too --
       the rubric rewards 'strong but not exhausted', 70+ is not that).
    1: 45 <= RSI <= 55.
    2: 55 < RSI <= 70.
    """
    r = c.rsi14
    if RSI_ZONE_1_MIN <= r <= RSI_ZONE_1_MAX:
        return 1, f"R2=1 RSI {r:.1f} in 45-55"
    if RSI_ZONE_1_MAX < r <= RSI_ZONE_2_MAX:
        return 2, f"R2=2 RSI {r:.1f} in 55-70"
    return 0, f"R2=0 RSI {r:.1f} outside 45-70"


def _score_r3(c: Candidate) -> tuple[int, str]:
    """Volume confirmation -- RVOL + OBV slope.
    0: RVOL < 2.
    1: 2 <= RVOL <= 3, OBV flat (or rising) -- 'flat' meaning not falling.
    2: RVOL > 3 AND OBV rising.
    """
    if c.rvol < RVOL_ZONE_1_MIN:
        return 0, f"R3=0 RVOL {c.rvol:.2f} < 2"
    if c.rvol > RVOL_ZONE_2_MIN and c.obv_slope > 0:
        return 2, f"R3=2 RVOL {c.rvol:.2f} > 3 and OBV rising"
    if c.obv_slope >= 0:
        return 1, f"R3=1 RVOL {c.rvol:.2f} in 2-3, OBV not falling"
    return 0, f"R3=0 OBV falling despite RVOL {c.rvol:.2f}"


def _score_r4(c: Candidate) -> tuple[int, str]:
    """Volatility quality -- ADX14 + ATR expansion.
    0: ADX < 15.
    1: 15 <= ADX <= 25.
    2: ADX > 25 AND ATR expanding.
    """
    if c.adx14 < ADX_ZONE_1_MIN:
        return 0, f"R4=0 ADX {c.adx14:.1f} < 15"
    if c.adx14 > ADX_ZONE_2_MIN and c.atr_expanding:
        return 2, f"R4=2 ADX {c.adx14:.1f} > 25 with ATR expanding"
    return 1, f"R4=1 ADX {c.adx14:.1f} in 15-25"


def _score_r5(c: Candidate) -> tuple[int, str]:
    """Structure -- position vs opening range / prior day high.
    0: price below prior close.
    1: price inside opening range.
    2: broke opening-range high AND held on retest.
    """
    if c.price < c.prior_close:
        return 0, f"R5=0 price {c.price:.2f} < prior close {c.prior_close:.2f}"
    if c.broke_opening_range_high and c.held_on_retest:
        return 2, f"R5=2 broke OR high and held on retest"
    if c.opening_range_low <= c.price <= c.opening_range_high:
        return 1, f"R5=1 inside opening range"
    # Above OR high but did not break-and-hold (e.g. broke but failed
    # the retest) OR above prior close but below OR low -- neither
    # scenario earns the top score; falls back to 1 (structure is
    # constructive but not confirmed).
    return 1, f"R5=1 above prior close, structure not confirmed"


def _score_r6(c: Candidate) -> tuple[int, str]:
    """Relative strength -- vs SPY (equities) or BTC (crypto), 30-min.
    0: underperforming benchmark (relative_strength_pct < 0).
    1: in line (0 <= relative_strength_pct <= 1).
    2: outperforming by > 1%.
    """
    rs = c.relative_strength_pct
    if rs < 0:
        return 0, f"R6=0 underperforming benchmark ({rs:+.2f}%)"
    if rs > RELATIVE_STRENGTH_OUTPERFORM_PCT:
        return 2, f"R6=2 outperforming benchmark by {rs:+.2f}%"
    return 1, f"R6=1 in line with benchmark ({rs:+.2f}%)"


_SCORERS = (_score_r1, _score_r2, _score_r3, _score_r4, _score_r5, _score_r6)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def score(candidate: Candidate) -> Score:
    """Score one candidate against the 6-component rubric. Emits the full
    per-component breakdown. `entry_eligible` folds the total-threshold
    AND the R1/R6 hard floors into one flag; `ineligible_reason` names
    which check failed when it's False, in the order the gate itself
    would care about (total first, then the hard floors).
    """
    per: list[tuple[int, str]] = [scorer(candidate) for scorer in _SCORERS]
    breakdown = {key: pts for key, (pts, _) in zip(COMPONENT_KEYS, per)}
    reasons = tuple(reason for _, reason in per)
    total = sum(breakdown.values())

    ineligible_reason: str | None = None
    if total < ENTRY_SCORE_THRESHOLD:
        ineligible_reason = "total_below_threshold"
    elif breakdown["R1"] < HARD_MIN_R1:
        ineligible_reason = "r1_below_floor"
    elif breakdown["R6"] < HARD_MIN_R6:
        ineligible_reason = "r6_below_floor"

    return Score(
        symbol=candidate.symbol,
        venue=candidate.venue,
        total=total,
        breakdown=breakdown,
        reasons=reasons,
        entry_eligible=ineligible_reason is None,
        ineligible_reason=ineligible_reason,
    )


def rank(candidates: list[Candidate]) -> list[Score]:
    """Score every candidate; return them sorted by (entry_eligible desc,
    total desc, symbol asc). Ineligible candidates aren't dropped -- the
    caller (M009 digest) still shows them, but eligible ones come first
    and the highest-scoring eligible one is the head. Ties break on
    symbol so the order is deterministic across runs.
    """
    scored = [score(c) for c in candidates]
    scored.sort(key=lambda s: (not s.entry_eligible, -s.total, s.symbol))
    return scored
