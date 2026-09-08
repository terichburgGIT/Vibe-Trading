"""M008 -- Metrics engine. Phase D, T018.

Implements every metric in `Metrics_Definitions.md` sections 1-2, plus
the rolling-scorecard summary in section 5. Pure math over a list of
closed cards (as returned by `vt.journal.store.closed_cards`); no
persistence, no I/O.

Design invariants (Metrics_Definitions.md sections 0, 3, 6):
  * Everything is measured in R, never dollars. Dollar P&L conflates
    "was this a good decision" with "how big was the bet" -- kept
    separate so the history is teachable.
  * Session numbers exist but are labelled *diagnostic*. Only the
    rolling window (default 100 trades) gets a verdict. Below the sample
    threshold, the scorecard says "n too low" rather than colouring the
    row green or red -- that's the T027 sample-size guard's role,
    surfaced here so it renders even in a plain terminal print.
  * The pathological case Metrics_Definitions.md § 2 flags explicitly
    -- 70% win rate with negative expectancy -- must NEVER render as
    healthy. `scorecard` verdicts on expectancy, not win rate, and this
    is pinned by an integration test in `test_journal_metrics.py`.

Full contract in `03_Modules.md` section M008.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


# Metrics_Definitions.md section 2 bars-to-clear.
EXPECTANCY_BAR = 0.10  # R
PROFIT_FACTOR_BAR = 1.30
SQN_BAR = 2.0
MAX_DRAWDOWN_BAR_R = -10.0  # more negative = worse
SORTINO_BAR = 1.0
PAYOFF_RATIO_BAR = 1.5
EDGE_RATIO_BAR = 1.5
CAPTURE_RATIO_BAR = 0.5
COST_RATIO_BAR = 0.10

# Metrics_Definitions.md section 3 sample-size thresholds.
MIN_N_FOR_PLUMBING = 20  # engineering validation only, NO edge conclusion
MIN_N_FOR_EXPECTANCY = 100  # M-8 gate: expectancy plausibly positive
MIN_N_FOR_SIGNIFICANCE = 400  # t > 2 at 0.10R / 1.0R


# --------------------------------------------------------------------------- #
# Per-trade extraction -- pull the R-multiple out of a closed card
# --------------------------------------------------------------------------- #


def _r(card: Mapping[str, Any]) -> float:
    outcome = card.get("outcome")
    if outcome is None:
        raise ValueError(f"card {card.get('card_id')!r} has no outcome yet")
    return float(outcome["r_multiple"])


def r_multiples(cards: Iterable[Mapping[str, Any]]) -> list[float]:
    return [_r(c) for c in cards]


# --------------------------------------------------------------------------- #
# Aggregate metrics -- Metrics_Definitions.md section 2
# --------------------------------------------------------------------------- #


def expectancy(cards: Sequence[Mapping[str, Any]]) -> float:
    """Metrics_Definitions.md section 2 headline: the mean R-multiple
    across all trades. Everything else is diagnostic. Empty input
    returns 0.0 (not NaN) so the scorecard can render early."""
    rs = r_multiples(cards)
    return statistics.fmean(rs) if rs else 0.0


def win_rate(cards: Sequence[Mapping[str, Any]]) -> float:
    """Wins / total. Metrics_Definitions.md § 2 warning: 'meaningless
    alone'. Track it, never optimize for it -- the strategy's intended
    win rate is 40-50%; a 65% reading is a red flag that exits are
    cutting winners short, not a cause for celebration."""
    rs = r_multiples(cards)
    return sum(1 for r in rs if r > 0) / len(rs) if rs else 0.0


def _split_win_loss(rs: Sequence[float]) -> tuple[list[float], list[float]]:
    return [r for r in rs if r > 0], [r for r in rs if r < 0]


def avg_win_r(cards: Sequence[Mapping[str, Any]]) -> float:
    wins, _ = _split_win_loss(r_multiples(cards))
    return statistics.fmean(wins) if wins else 0.0


def avg_loss_r(cards: Sequence[Mapping[str, Any]]) -> float:
    """Returned as a POSITIVE number (magnitude of the average losing
    R). This matches the payoff-ratio formula 'avgWin / avgLoss' where
    both operands are positive."""
    _, losses = _split_win_loss(r_multiples(cards))
    return -statistics.fmean(losses) if losses else 0.0


def payoff_ratio(cards: Sequence[Mapping[str, Any]]) -> float:
    """avgWin_R / avgLoss_R. Zero-guard for a no-loss window (a system
    with 0 losers yet cannot be assessed by payoff ratio -- returns
    inf, and the scorecard handles that separately)."""
    aw = avg_win_r(cards)
    al = avg_loss_r(cards)
    if al == 0:
        return math.inf if aw > 0 else 0.0
    return aw / al


def profit_factor(cards: Sequence[Mapping[str, Any]]) -> float:
    """gross wins / gross losses. Same zero-guard shape as payoff."""
    wins, losses = _split_win_loss(r_multiples(cards))
    gross_wins = sum(wins)
    gross_losses = -sum(losses)
    if gross_losses == 0:
        return math.inf if gross_wins > 0 else 0.0
    return gross_wins / gross_losses


def sqn(cards: Sequence[Mapping[str, Any]]) -> float:
    """Van Tharp's System Quality Number: sqrt(n) * mean_R / stdev_R.
    A t-statistic in disguise. n < 2 or zero stdev -> 0.0, since the
    formula is undefined there and the scorecard's 'n too low' banner
    is a more honest surface than a synthetic value."""
    rs = r_multiples(cards)
    n = len(rs)
    if n < 2:
        return 0.0
    sd = statistics.stdev(rs)
    if sd == 0:
        return 0.0
    return math.sqrt(n) * statistics.fmean(rs) / sd


def max_drawdown_r(cards: Sequence[Mapping[str, Any]]) -> float:
    """Peak-to-trough R, as a signed number (0 or negative). Cumulative
    R runs by insertion order (the journal preserves that -- see
    store.read_cards). A single winning day returns 0.0."""
    rs = r_multiples(cards)
    peak = 0.0
    cum = 0.0
    worst = 0.0
    for r in rs:
        cum += r
        if cum > peak:
            peak = cum
        drawdown = cum - peak
        if drawdown < worst:
            worst = drawdown
    return worst


def sortino_ratio(cards: Sequence[Mapping[str, Any]]) -> float:
    """excess return / downside deviation. Preferred over Sharpe
    (Metrics_Definitions.md § 2) because Sharpe penalizes upside
    volatility, which is the volatility you want. 'Excess return' here
    is the mean R (the risk-free R-multiple for a per-trade series is
    0 -- you didn't have to take the trade). Downside deviation uses
    only the negative-R trades; zero downside means infinite Sortino,
    which is what the scorecard's zero-loss branch surfaces."""
    rs = r_multiples(cards)
    if not rs:
        return 0.0
    mean = statistics.fmean(rs)
    downside = [r for r in rs if r < 0]
    if not downside:
        return math.inf if mean > 0 else 0.0
    # Root-mean-square of the negative excursions (relative to 0).
    downside_dev = math.sqrt(sum(r * r for r in downside) / len(downside))
    if downside_dev == 0:
        return 0.0
    return mean / downside_dev


def cumulative_r(cards: Sequence[Mapping[str, Any]]) -> float:
    return sum(r_multiples(cards))


# --------------------------------------------------------------------------- #
# MAE / MFE aggregates (Metrics_Definitions.md section 1)
# --------------------------------------------------------------------------- #


def mean_mae(cards: Sequence[Mapping[str, Any]], *, winners_only: bool = False) -> float:
    """Mean Max Adverse Excursion in R. `winners_only=True` restricts
    to winning trades -- Metrics_Definitions.md § 1 note: 'winners
    should have low MAE'; if winners rarely exceed 0.5R MAE while the
    stop sits at 1.0R, the stop is too wide."""
    values: list[float] = []
    for c in cards:
        outcome = c.get("outcome") or {}
        if winners_only and float(outcome.get("r_multiple", 0)) <= 0:
            continue
        mae = outcome.get("mae")
        if mae is None:
            continue
        values.append(float(mae))
    return statistics.fmean(values) if values else 0.0


def mean_mfe(cards: Sequence[Mapping[str, Any]]) -> float:
    values = [
        float(c["outcome"]["mfe"])
        for c in cards
        if c.get("outcome") and c["outcome"].get("mfe") is not None
    ]
    return statistics.fmean(values) if values else 0.0


def edge_ratio(cards: Sequence[Mapping[str, Any]]) -> float:
    """MFE / MAE across the population. Metrics_Definitions.md § 1
    target: > 1.5 (the setup gives more than it takes). Uses absolute
    MAE since MAE is stored as a signed excursion (negative R)."""
    mae = abs(mean_mae(cards))
    mfe = mean_mfe(cards)
    if mae == 0:
        return math.inf if mfe > 0 else 0.0
    return mfe / mae


def capture_ratio(cards: Sequence[Mapping[str, Any]]) -> float:
    """realized R / MFE, averaged over trades where MFE > 0. What
    fraction of the available move the exit rules actually caught.
    Section 1 target: > 0.5."""
    ratios: list[float] = []
    for c in cards:
        outcome = c.get("outcome") or {}
        mfe = outcome.get("mfe")
        if mfe is None or float(mfe) <= 0:
            continue
        ratios.append(float(outcome["r_multiple"]) / float(mfe))
    return statistics.fmean(ratios) if ratios else 0.0


def adherence_rate(cards: Sequence[Mapping[str, Any]]) -> float:
    """Fraction of trades tagged as rule-following. Cards without an
    explicit `adherence` field default to True (unlabelled is treated
    as followed-the-rule, since an override is meant to be tagged
    explicitly)."""
    if not cards:
        return 1.0
    followed = sum(1 for c in cards if (c.get("outcome") or {}).get("adherence", True))
    return followed / len(cards)


def override_expectancy(cards: Sequence[Mapping[str, Any]]) -> float:
    """Expectancy on ONLY the trades tagged as overrides (adherence=False).
    Metrics_Definitions.md § 1 makes this a first-class metric: 'the most
    useful number a discretionary trader can own.'"""
    overrides = [
        c for c in cards
        if (c.get("outcome") or {}).get("adherence", True) is False
    ]
    return expectancy(overrides)


# --------------------------------------------------------------------------- #
# Scorecard -- Metrics_Definitions.md section 5
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MetricLine:
    """One row of the scorecard. `verdict` is one of:
      * "pass"        -- meets the bar
      * "fail"        -- below the bar and n is large enough to say
      * "insufficient_n" -- below the sample-size threshold; render as
                            grey with an 'n too low' badge instead of
                            colouring red or green (T027 guard, per
                            Metrics_Definitions.md § 3).
      * "diagnostic"  -- not gated at all (e.g. win rate, per § 2).
    """

    name: str
    value: float
    bar: float | None
    verdict: str


@dataclass(frozen=True)
class Scorecard:
    """The full rolling-window report card. `n` is the number of closed
    trades in the window. `honest` is False iff n < MIN_N_FOR_EXPECTANCY
    -- the whole dashboard should be styled greyed-out in that case, no
    green ticks, no encouragement to size up.
    """

    n: int
    honest: bool
    lines: tuple[MetricLine, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def get(self, name: str) -> MetricLine:
        for line in self.lines:
            if line.name == name:
                return line
        raise KeyError(name)


def _verdict(value: float, bar: float, *, higher_is_better: bool, n: int, n_bar: int) -> str:
    if n < n_bar:
        return "insufficient_n"
    if higher_is_better:
        return "pass" if value >= bar else "fail"
    return "pass" if value <= bar else "fail"


def scorecard(
    cards: Sequence[Mapping[str, Any]],
    *,
    n_bar: int = MIN_N_FOR_EXPECTANCY,
) -> Scorecard:
    """Build the rolling scorecard. Metrics_Definitions.md § 5 layout,
    everything driven from the closed-cards list this receives. The
    caller is expected to already have windowed the input (e.g. last
    100 trades) -- this function doesn't slice.
    """
    n = len(cards)
    exp = expectancy(cards)
    pf = profit_factor(cards)
    s = sqn(cards)
    dd = max_drawdown_r(cards)
    wr = win_rate(cards)
    payoff = payoff_ratio(cards)
    sortino = sortino_ratio(cards)
    adherence = adherence_rate(cards)
    ovr_exp = override_expectancy(cards)

    lines: list[MetricLine] = [
        MetricLine("expectancy_r", exp, EXPECTANCY_BAR,
                   _verdict(exp, EXPECTANCY_BAR, higher_is_better=True, n=n, n_bar=n_bar)),
        MetricLine("profit_factor", pf, PROFIT_FACTOR_BAR,
                   _verdict(pf, PROFIT_FACTOR_BAR, higher_is_better=True, n=n, n_bar=n_bar)),
        MetricLine("sqn", s, SQN_BAR,
                   _verdict(s, SQN_BAR, higher_is_better=True, n=n, n_bar=n_bar)),
        MetricLine("max_drawdown_r", dd, MAX_DRAWDOWN_BAR_R,
                   _verdict(dd, MAX_DRAWDOWN_BAR_R, higher_is_better=True, n=n, n_bar=n_bar)),
        MetricLine("sortino", sortino, SORTINO_BAR,
                   _verdict(sortino, SORTINO_BAR, higher_is_better=True, n=n, n_bar=n_bar)),
        MetricLine("payoff_ratio", payoff, PAYOFF_RATIO_BAR,
                   _verdict(payoff, PAYOFF_RATIO_BAR, higher_is_better=True, n=n, n_bar=n_bar)),
        MetricLine("win_rate", wr, None, "diagnostic"),
        MetricLine("adherence_rate", adherence, 1.0,
                   _verdict(adherence, 0.95, higher_is_better=True, n=n, n_bar=n_bar)),
        MetricLine("override_expectancy_r", ovr_exp, None, "diagnostic"),
    ]

    warnings: list[str] = []
    if n < MIN_N_FOR_PLUMBING:
        warnings.append(f"n={n}: below plumbing-check threshold ({MIN_N_FOR_PLUMBING})")
    if n < n_bar:
        warnings.append(
            f"n={n}: below expectancy-verdict threshold ({n_bar}); "
            "every verdict shown as insufficient_n per Metrics_Definitions.md section 3"
        )
    # The pathological case Metrics_Definitions.md § 2 calls out.
    if wr > 0.6 and exp < 0:
        warnings.append(
            f"win_rate={wr:.0%} with negative expectancy ({exp:+.3f}R) -- "
            "the classic trap: feels good, loses money"
        )

    return Scorecard(
        n=n,
        honest=n >= n_bar,
        lines=tuple(lines),
        warnings=tuple(warnings),
    )
