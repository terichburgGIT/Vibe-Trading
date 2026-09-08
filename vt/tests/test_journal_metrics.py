"""Tests for M008 -- vt.journal.metrics (T018; see 06_Tests.md).

Phase D (`16_Next_Steps.md`): T018 -- metrics math.
Fixture of 100 known trades. Expectancy, profit factor, SQN, MAE/MFE,
Sortino, max DD all match hand-computed values. Includes the
pathological case: 70% win rate with negative expectancy -- asserts
the dashboard does not render that as healthy.

Small hand-verifiable fixtures are used for the per-metric tests; the
100-trade case appears once, as the integration check that everything
adds up together and that the scorecard passes the healthy-looking-but-
losing example correctly.

Run with: pytest vt/tests -m unit
"""

from __future__ import annotations

import math
from statistics import fmean, stdev

import pytest

from vt.journal import metrics

pytestmark = pytest.mark.unit


def _card(r: float, **outcome_overrides) -> dict:
    outcome = {"r_multiple": r, "exit_reason": "target_1", "adherence": True}
    outcome.update(outcome_overrides)
    return {"card_id": f"VT-{id(outcome)}", "outcome": outcome}


# --------------------------------------------------------------------------- #
# Expectancy / win rate / payoff -- hand-computed fixtures
# --------------------------------------------------------------------------- #


def test_expectancy_is_mean_of_r_multiples() -> None:
    rs = [1.5, -1.0, 2.0, -1.0, 0.5]
    cards = [_card(r) for r in rs]
    assert metrics.expectancy(cards) == pytest.approx(sum(rs) / len(rs))


def test_win_rate_wins_over_total() -> None:
    cards = [_card(1.0), _card(-1.0), _card(2.0), _card(-1.0)]
    assert metrics.win_rate(cards) == pytest.approx(0.5)


def test_avg_loss_is_returned_as_positive_magnitude() -> None:
    cards = [_card(-1.0), _card(-2.0), _card(3.0)]
    assert metrics.avg_loss_r(cards) == pytest.approx(1.5)
    assert metrics.avg_win_r(cards) == pytest.approx(3.0)


def test_payoff_ratio_is_avg_win_over_avg_loss() -> None:
    cards = [_card(3.0), _card(-1.0), _card(-1.0)]
    # avg_win = 3.0, avg_loss = 1.0 -> payoff = 3.0
    assert metrics.payoff_ratio(cards) == pytest.approx(3.0)


def test_payoff_ratio_is_inf_when_no_losses_but_wins_present() -> None:
    cards = [_card(1.0), _card(2.0)]
    assert math.isinf(metrics.payoff_ratio(cards))


def test_profit_factor_gross_wins_over_gross_losses() -> None:
    cards = [_card(3.0), _card(1.0), _card(-1.0), _card(-1.0)]
    # gross_wins=4, gross_losses=2 -> PF=2.0
    assert metrics.profit_factor(cards) == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# SQN -- Van Tharp's system-quality number
# --------------------------------------------------------------------------- #


def test_sqn_matches_sqrt_n_times_mean_over_stdev() -> None:
    rs = [1.5, -1.0, 2.0, -1.0, 0.5, 1.0, -0.5, 1.2]
    cards = [_card(r) for r in rs]
    expected = math.sqrt(len(rs)) * fmean(rs) / stdev(rs)
    assert metrics.sqn(cards) == pytest.approx(expected)


def test_sqn_is_zero_when_n_below_two() -> None:
    assert metrics.sqn([]) == 0.0
    assert metrics.sqn([_card(1.0)]) == 0.0


def test_sqn_is_zero_when_all_trades_identical_stdev_is_zero() -> None:
    cards = [_card(0.5) for _ in range(5)]
    assert metrics.sqn(cards) == 0.0


# --------------------------------------------------------------------------- #
# Max drawdown -- cumulative peak-to-trough in R
# --------------------------------------------------------------------------- #


def test_max_drawdown_r_walks_cumulative_and_returns_signed_worst() -> None:
    # Cumulative: 1, -1, 2, 0, -3, -1
    # Peak so far: 1, 1, 2, 2, 2, 2
    # Drawdown:    0, -2, 0, -2, -5, -3
    # Worst: -5
    rs = [1.0, -2.0, 3.0, -2.0, -3.0, 2.0]
    cards = [_card(r) for r in rs]
    assert metrics.max_drawdown_r(cards) == pytest.approx(-5.0)


def test_max_drawdown_r_is_zero_when_only_winners() -> None:
    cards = [_card(r) for r in (1.0, 0.5, 2.0)]
    assert metrics.max_drawdown_r(cards) == 0.0


# --------------------------------------------------------------------------- #
# MAE / MFE / edge ratio / capture ratio
# --------------------------------------------------------------------------- #


def test_mean_mae_averages_only_cards_with_mae_field() -> None:
    cards = [_card(1.0, mae=-0.3), _card(-1.0, mae=-1.0), _card(1.5)]  # last has no mae
    assert metrics.mean_mae(cards) == pytest.approx(-0.65)


def test_mean_mae_winners_only_isolates_winning_trades() -> None:
    cards = [_card(1.0, mae=-0.3), _card(2.0, mae=-0.1), _card(-1.0, mae=-1.0)]
    assert metrics.mean_mae(cards, winners_only=True) == pytest.approx(-0.2)


def test_edge_ratio_is_mfe_over_absolute_mae() -> None:
    cards = [_card(1.0, mae=-0.5, mfe=1.5), _card(-1.0, mae=-1.0, mfe=0.5)]
    # mean_mae = -0.75, mean_mfe = 1.0 -> edge = 1.0 / 0.75
    assert metrics.edge_ratio(cards) == pytest.approx(1.0 / 0.75)


def test_capture_ratio_averages_realized_over_mfe_for_cards_with_positive_mfe() -> None:
    # Card A: r=1.0, mfe=2.0 -> capture 0.5
    # Card B: r=0.4, mfe=0.8 -> capture 0.5
    # Card C: r=-1.0, mfe=0 -> skipped
    cards = [
        _card(1.0, mfe=2.0),
        _card(0.4, mfe=0.8),
        _card(-1.0, mfe=0.0),
    ]
    assert metrics.capture_ratio(cards) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Sortino -- excess return over downside deviation
# --------------------------------------------------------------------------- #


def test_sortino_uses_only_negative_returns_in_the_denominator() -> None:
    rs = [1.0, 2.0, -1.0, -2.0, 0.5]
    cards = [_card(r) for r in rs]
    mean = fmean(rs)
    downside = [r for r in rs if r < 0]
    dd = math.sqrt(sum(r * r for r in downside) / len(downside))
    assert metrics.sortino_ratio(cards) == pytest.approx(mean / dd)


def test_sortino_is_inf_when_mean_positive_and_no_downside() -> None:
    cards = [_card(r) for r in (1.0, 2.0, 0.5)]
    assert math.isinf(metrics.sortino_ratio(cards))


# --------------------------------------------------------------------------- #
# Adherence
# --------------------------------------------------------------------------- #


def test_adherence_rate_defaults_to_true_when_unlabelled() -> None:
    cards = [
        _card(1.0),  # implicit adherence=True
        _card(-1.0, adherence=False),
        _card(0.5, adherence=True),
    ]
    assert metrics.adherence_rate(cards) == pytest.approx(2 / 3)


def test_override_expectancy_isolates_override_trades() -> None:
    cards = [
        _card(1.0, adherence=True),
        _card(-2.0, adherence=False),
        _card(-1.0, adherence=False),
        _card(0.5, adherence=True),
    ]
    assert metrics.override_expectancy(cards) == pytest.approx(-1.5)


# --------------------------------------------------------------------------- #
# Scorecard -- verdicts, insufficient_n gating, the T018 pathological case
# --------------------------------------------------------------------------- #


def test_scorecard_flags_insufficient_n_below_the_bar() -> None:
    cards = [_card(1.0), _card(-0.5), _card(1.2), _card(-0.8), _card(0.9)]  # n=5
    sc = metrics.scorecard(cards)
    assert sc.n == 5
    assert sc.honest is False
    # Everything gated on n should read insufficient_n.
    for name in ("expectancy_r", "profit_factor", "sqn", "max_drawdown_r", "sortino"):
        assert sc.get(name).verdict == "insufficient_n", name


def test_scorecard_gives_verdicts_when_n_meets_the_bar() -> None:
    """Clean-edge fixture: 5 winners @ +2.0R, 5 losers @ -1.0R.
    Mean = +0.5R, gross_wins/gross_losses = 10/5 = 2.0. Both clear
    the 0.10R expectancy and 1.30 profit-factor bars once n is at
    threshold. n_bar lowered here so the test is deterministic
    without needing 100 real cards."""
    rs = [2.0, 2.0, 2.0, 2.0, 2.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    cards = [_card(r) for r in rs]
    sc = metrics.scorecard(cards, n_bar=len(cards))
    assert sc.honest is True
    assert sc.get("expectancy_r").verdict == "pass"
    assert sc.get("expectancy_r").value == pytest.approx(0.5)
    assert sc.get("profit_factor").verdict == "pass"
    assert sc.get("profit_factor").value == pytest.approx(2.0)
    # win_rate is always diagnostic, never pass/fail.
    assert sc.get("win_rate").verdict == "diagnostic"


def test_scorecard_pathological_high_win_rate_negative_expectancy_does_not_render_healthy() -> None:
    """Metrics_Definitions.md § 2 headline pathology: 70% win rate with
    negative expectancy. The dashboard MUST NOT render this as green --
    the expectancy line must fail, and there must be a warning surfacing
    the exact trap. This is the assertion T018 was written for."""
    # 7 winners of +0.3R, 3 losers of -1.0R -> win_rate 70%, mean = (2.1-3.0)/10 = -0.09
    rs = [0.3] * 7 + [-1.0] * 3
    cards = [_card(r) for r in rs]
    sc = metrics.scorecard(cards, n_bar=len(cards))

    assert metrics.win_rate(cards) == pytest.approx(0.7)
    assert metrics.expectancy(cards) < 0
    exp_line = sc.get("expectancy_r")
    assert exp_line.verdict == "fail"
    assert any(
        "win_rate" in w and "negative expectancy" in w for w in sc.warnings
    ), f"expected the healthy-looking-but-losing warning; got {sc.warnings!r}"


def test_scorecard_hundred_trade_fixture_integration() -> None:
    """The T018 spec fixture: 100 trades, everything computed matches
    hand-computed aggregates. Construction: 45 winners of +2.0R, 55
    losers of -1.0R. Deliberately synthetic so every aggregate has a
    closed-form check."""
    rs = [2.0] * 45 + [-1.0] * 55
    cards = [_card(r) for r in rs]

    assert metrics.expectancy(cards) == pytest.approx(0.35)  # (90 - 55) / 100
    assert metrics.win_rate(cards) == pytest.approx(0.45)
    assert metrics.avg_win_r(cards) == pytest.approx(2.0)
    assert metrics.avg_loss_r(cards) == pytest.approx(1.0)
    assert metrics.payoff_ratio(cards) == pytest.approx(2.0)
    # gross_wins=90, gross_losses=55 -> PF=90/55
    assert metrics.profit_factor(cards) == pytest.approx(90 / 55)
    # cumulative walk: winners then losers -> peak 90, trough at end 35
    # But actual walk: +2, +4, ... peak at cum=90, then -1 each -> min = 90-55 = 35
    # dd = 35 - 90 = -55.
    assert metrics.max_drawdown_r(cards) == pytest.approx(-55.0)

    sc = metrics.scorecard(cards, n_bar=100)
    assert sc.honest is True
    assert sc.get("expectancy_r").verdict == "pass"
    assert sc.get("profit_factor").verdict == "pass"
    # Payoff ratio 2.0 clears the 1.5 bar.
    assert sc.get("payoff_ratio").verdict == "pass"
    # Max DD -55R fails the -10R bar spectacularly -- the fixture has a
    # brutal shape by construction (all wins then all losses) so this
    # asserts the DD math is directional-signed correctly.
    assert sc.get("max_drawdown_r").verdict == "fail"
