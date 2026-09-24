"""M010 -- Walk-forward no-lookahead assertion (T020). Phase E.

`06_Tests.md` T020: "No indicator at bar t uses data from t+1. Verified
by feeding future bars as NaN -- results must be unchanged." This
module is that verification, applied to M003's `vt.indicators.engine.
compute` -- the function every rubric score (M005) is ultimately built
from, and the one `vt/indicators/engine.py`'s own module docstring
already claims is "pure forward-only functions (no lookahead ... matters
for M010's T020 later)" (S012). T020 is what actually tests that claim
rather than trusting it.

Method: for every bar index i, build a copy of the bar series where
every bar AFTER i has its OHLCV fields replaced with NaN (`time`,
`symbol`, `source_feed` untouched -- VWAP's session-reset and the
crypto/equity branch both read those, and this test isn't about them).
Run `compute()` on that copy and compare its value at index i against
the value `compute()` produced on the real, untouched series at the
same index. Any difference -- including a NaN appearing where a real
number was -- means some future bar's data reached backward into an
earlier bar's computed value. Lookahead bias is called out in T020's
own spec as "the defect most likely to make a worthless strategy look
brilliant": a backtest that can see tomorrow's close will always find
an edge, and that edge will evaporate the instant it trades live.

This deliberately checks M003's `compute()` directly rather than a full
simulated-trading walk-forward loop (M002 screen -> M005 rubric -> M006
gate over history) -- that fuller harness is real M010 scope (`03_
Modules.md` M010) but is a separably large piece of work, not yet
built. Lookahead in the indicator layer is the specific, high-value
defect T020 names, and it is fully exercisable today against the real
M003 code.

Full contract in `03_Modules.md` section M010; test spec in
`06_Tests.md` T020.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Callable, Sequence

from vt.data.feed import Bar
from vt.indicators.engine import IndicatorFrame, compute

#: Fields an IndicatorFrame carries, aligned 1:1 with the input bars --
#: see `IndicatorFrame`'s own docstring for the meaning of `None`.
_FRAME_FIELDS = ("vwap", "ema9", "ema21", "rsi14", "obv", "adx14", "atr14")


class LookaheadDetectedError(RuntimeError):
    """Raised by `assert_no_lookahead` when `find_lookahead_violations`
    returns at least one violation. Carries the count and the first
    violation in the message so a failing walk-forward run fails loud
    with a pointer to where to look, not just "something's wrong."
    """


@dataclass(frozen=True)
class LookaheadViolation:
    """One (bar index, field) pair where masking every bar after that
    index changed the computed value at that index -- proof some future
    bar's data was read to produce it.
    """

    bar_index: int
    field: str
    value_with_full_data: float | None
    value_with_future_masked: float | None


def bars_with_future_nan(bars: Sequence[Bar], as_of_index: int) -> list[Bar]:
    """Copy of `bars` where every bar with index > `as_of_index` has its
    OHLCV fields replaced with NaN. `time`/`symbol`/`source_feed` are
    left untouched -- masking those would break VWAP's session-boundary
    read and the crypto/equity branch, which is not what this test is
    checking. `Bar` is a frozen dataclass, so this builds new instances
    via `dataclasses.replace` rather than mutating.
    """
    if as_of_index < 0 or as_of_index >= len(bars):
        raise ValueError(f"as_of_index must be within [0, {len(bars) - 1}], got {as_of_index}")

    nan = float("nan")
    return [
        bar if i <= as_of_index else dataclasses.replace(bar, open=nan, high=nan, low=nan, close=nan, volume=nan)
        for i, bar in enumerate(bars)
    ]


def _values_differ(full_value: float | None, masked_value: float | None) -> bool:
    if full_value is None or masked_value is None:
        return full_value is not masked_value
    full_is_nan = isinstance(full_value, float) and math.isnan(full_value)
    masked_is_nan = isinstance(masked_value, float) and math.isnan(masked_value)
    if full_is_nan or masked_is_nan:
        return not (full_is_nan and masked_is_nan)
    return full_value != masked_value


def find_lookahead_violations(
    bars: Sequence[Bar], compute_fn: Callable[[Sequence[Bar]], IndicatorFrame] = compute
) -> list[LookaheadViolation]:
    """Run `compute_fn` once on the real, untouched `bars`, then once per
    bar index on a future-masked copy, diffing every field at the
    boundary index each time. Returns every violation found; an empty
    list means no lookahead was detected. `compute_fn` defaults to the
    real M003 `compute`, but is swappable so a test can prove this
    detector actually catches a *known* lookahead bug, not just agrees
    with already-correct code (see `test_walk_forward.py`'s mirror
    test).
    """
    if not bars:
        return []

    full_frame = compute_fn(bars)
    violations: list[LookaheadViolation] = []

    for as_of_index in range(len(bars)):
        masked_bars = bars_with_future_nan(bars, as_of_index)
        masked_frame = compute_fn(masked_bars)

        for field in _FRAME_FIELDS:
            full_value = getattr(full_frame, field)[as_of_index]
            masked_value = getattr(masked_frame, field)[as_of_index]
            if _values_differ(full_value, masked_value):
                violations.append(
                    LookaheadViolation(
                        bar_index=as_of_index,
                        field=field,
                        value_with_full_data=full_value,
                        value_with_future_masked=masked_value,
                    )
                )

    return violations


def assert_no_lookahead(bars: Sequence[Bar], compute_fn: Callable[[Sequence[Bar]], IndicatorFrame] = compute) -> None:
    violations = find_lookahead_violations(bars, compute_fn)
    if violations:
        first = violations[0]
        raise LookaheadDetectedError(
            f"{len(violations)} lookahead violation(s) found in {compute_fn!r}; first at "
            f"bar_index={first.bar_index} field={first.field!r}: "
            f"full_data={first.value_with_full_data!r} vs future_masked={first.value_with_future_masked!r}"
        )
