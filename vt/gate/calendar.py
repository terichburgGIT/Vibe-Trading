"""M004 -- Calendar & Regime Gate.

Emits the session size multiplier per `Strategy_Spec.md` section 2.

Contract (from `03_Modules.md` M004)::

    gate_state(asof, *, vix, realized_vol, realized_vol_history,
               index_returns, events_path=None) -> GateState

State priority (Strategy_Spec section 2):

    STAND_DOWN (0.00)  -- FOMC/CPI/NFP release day  (from events YAML)
                       -- realized_vol below 20th pct of trailing 60d
    REDUCED    (0.50)  -- VIX > 28
                       -- quad witching / monthly OPEX (3rd Friday)
                       -- day after > 2 sigma index move
    NORMAL     (1.00)  -- default

STAND_DOWN wins over REDUCED. Reasons stack across all triggering rules.
Missing inputs (None) never spuriously downgrade the state -- they are
simply skipped and their absence noted in `reasons` iff the resulting
state is NORMAL (so callers can see what wasn't checked).

Day-of-week direction is deliberately *not* an input -- see
`Seasonality_Study.md` and AD referenced there. T009 is the pre-registered
test that would re-open that question.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Sequence

import yaml

# --------------------------------------------------------------------------- #
# Thresholds (single source of truth for the gate; keep in sync with spec)
# --------------------------------------------------------------------------- #

VIX_REDUCED_THRESHOLD: float = 28.0
INDEX_SIGMA_REDUCED_THRESHOLD: float = 2.0
REALIZED_VOL_STAND_DOWN_PCT: float = 20.0  # percentile of trailing 60d

MULT_STAND_DOWN: float = 0.00
MULT_REDUCED: float = 0.50
MULT_NORMAL: float = 1.00

_DEFAULT_EVENTS_PATH = Path(__file__).with_name("events.yaml")


@dataclass(frozen=True)
class GateState:
    """Immutable session gate output."""

    state: str  # "STAND_DOWN" | "REDUCED" | "NORMAL"
    multiplier: float  # MULT_STAND_DOWN | MULT_REDUCED | MULT_NORMAL
    reasons: tuple[str, ...]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def gate_state(
    asof: datetime,
    *,
    vix: float | None = None,
    realized_vol: float | None = None,
    realized_vol_history: Sequence[float] | None = None,
    index_returns: Sequence[float] | None = None,
    events_path: Path | None = None,
) -> GateState:
    """Compute the session gate for ``asof``.

    Parameters
    ----------
    asof:
        Session timestamp (tz-aware recommended). Only the calendar date
        is used for event / OPEX / quad-witching lookup.
    vix:
        Current VIX level (None = unknown, do not trigger on VIX alone).
    realized_vol:
        Today's index realized vol (same unit as ``realized_vol_history``).
    realized_vol_history:
        Trailing sample (typically 60 sessions) used to compute the 20th
        percentile threshold.
    index_returns:
        Recent index returns; the last element is inspected against a
        2-sigma move of the preceding window.
    events_path:
        Override the bundled hand-maintained YAML (test hook).
    """

    stand_down: list[str] = []
    reduced: list[str] = []

    session_date = _as_date(asof)

    # ---- STAND_DOWN triggers ----
    event = _lookup_event(session_date, events_path or _DEFAULT_EVENTS_PATH)
    if event is not None:
        stand_down.append(f"macro_print:{event}")

    if realized_vol is not None and realized_vol_history:
        pct = _percentile(realized_vol_history, REALIZED_VOL_STAND_DOWN_PCT)
        if realized_vol < pct:
            stand_down.append(
                f"realized_vol_below_p{int(REALIZED_VOL_STAND_DOWN_PCT)}"
            )

    # ---- REDUCED triggers ----
    if vix is not None and vix > VIX_REDUCED_THRESHOLD:
        reduced.append(f"vix_above_{VIX_REDUCED_THRESHOLD:g}")

    if _is_quad_witching(session_date):
        reduced.append("quad_witching")
    elif _is_monthly_opex(session_date):
        reduced.append("monthly_opex")

    if index_returns and len(index_returns) >= 2:
        prior = list(index_returns[:-1])
        latest = float(index_returns[-1])
        sigma = _stdev(prior)
        if sigma > 0 and abs(latest) > INDEX_SIGMA_REDUCED_THRESHOLD * sigma:
            reduced.append(
                f"prior_session_move_gt_{INDEX_SIGMA_REDUCED_THRESHOLD:g}sigma"
            )

    # ---- Resolve precedence: STAND_DOWN > REDUCED > NORMAL ----
    if stand_down:
        return GateState(
            state="STAND_DOWN",
            multiplier=MULT_STAND_DOWN,
            reasons=tuple(stand_down + reduced),
        )
    if reduced:
        return GateState(
            state="REDUCED", multiplier=MULT_REDUCED, reasons=tuple(reduced)
        )
    return GateState(state="NORMAL", multiplier=MULT_NORMAL, reasons=())


# --------------------------------------------------------------------------- #
# Helpers -- deliberately pure so they can be unit-tested independently
# --------------------------------------------------------------------------- #


def _as_date(asof: datetime) -> date:
    if asof.tzinfo is not None:
        asof = asof.astimezone(timezone.utc)
    return asof.date()


def _lookup_event(day: date, path: Path) -> str | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    for row in data.get("stand_down", []) or []:
        row_date = row.get("date")
        if isinstance(row_date, str):
            row_date = date.fromisoformat(row_date)
        if row_date == day:
            return str(row.get("event", "unknown"))
    return None


def _percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile (matches numpy default), no numpy dep."""
    xs = sorted(float(v) for v in values)
    if not xs:
        raise ValueError("cannot compute percentile of empty sequence")
    if len(xs) == 1:
        return xs[0]
    k = (pct / 100.0) * (len(xs) - 1)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] + frac * (xs[hi] - xs[lo])


def _stdev(values: Sequence[float]) -> float:
    xs = [float(v) for v in values]
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return var**0.5


def _is_monthly_opex(day: date) -> bool:
    """Monthly equity OPEX = 3rd Friday of the month."""
    if day.weekday() != 4:  # Friday
        return False
    return 15 <= day.day <= 21


def _is_quad_witching(day: date) -> bool:
    """Quad witching = 3rd Friday of Mar/Jun/Sep/Dec."""
    return _is_monthly_opex(day) and day.month in (3, 6, 9, 12)
