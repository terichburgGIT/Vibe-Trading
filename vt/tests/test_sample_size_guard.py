"""T027 -- Sample-size guard: the test against our own future self-deception.

Metrics_Definitions.md § 3 is blunt: a single 8-hour session (3-8 trades)
carries, to a good approximation, *no* information about edge, and the
temptation to over-read the numbers peaks at the exact worst moment -- a
good week. This suite pins the guard that stops that, on both sides of the
seam:

  * the metrics layer (`vt.journal.metrics.scorecard`) must WITHHOLD a
    verdict on every gated metric below its required n, no matter how
    good the numbers look; and
  * the dashboard (`vt.render.card.render_digest`) must SHOW that
    withholding honestly -- "n too low" instead of a green check, and
    session-level (diagnostic) stats always labelled "diagnostic -- not
    evidence", never a verdict.

The canonical scenario here is a *flawless small sample*: eight trades,
all winners, huge expectancy, 100% win rate. If any green verdict
survives that, the guard is broken.

Run with: pytest vt/tests -m unit
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from vt.journal import metrics
from vt.render.card import render_digest

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fixtures -- the "good week" that must not fool the dashboard
# --------------------------------------------------------------------------- #

# Metrics that get a real pass/fail verdict once n is large enough. win_rate
# and override_expectancy_r are deliberately excluded: they are diagnostic by
# design (§ 2 -- "track it, never optimize"), never verdicted.
_GATED = (
    "expectancy_r",
    "profit_factor",
    "sqn",
    "max_drawdown_r",
    "sortino",
    "payoff_ratio",
    "adherence_rate",
)


def _winners(n: int, r: float = 2.0) -> list[dict]:
    """n all-winning trades -- a flawless (and statistically meaningless)
    small sample."""
    return [
        {"card_id": f"W{i}", "outcome": {"r_multiple": r + 0.01 * i, "adherence": True}}
        for i in range(n)
    ]


def _digest_html(sc: metrics.Scorecard) -> str:
    """Drive the real end-to-end dashboard path with a given scorecard."""
    return render_digest(
        {
            "date": "2026-09-15",
            "gate": {"state": "NORMAL"},
            "cards": [],
            "rejected": [],
            "scorecard": asdict(sc),
        }
    )


# --------------------------------------------------------------------------- #
# Metrics side -- the guard withholds a verdict below required n
# --------------------------------------------------------------------------- #


def test_a_flawless_small_sample_earns_no_green_verdict() -> None:
    """Eight winners in a row, +2R expectancy, 100% win rate -- and NOT a
    single gated metric is allowed to read `pass`. This is the whole point
    of the guard: a good week is not evidence."""
    sc = metrics.scorecard(_winners(8))  # default n_bar = MIN_N_FOR_EXPECTANCY (100)

    assert sc.honest is False
    assert metrics.win_rate(_winners(8)) == 1.0  # the numbers really are "great"
    assert metrics.expectancy(_winners(8)) > 0
    for name in _GATED:
        assert sc.get(name).verdict == "insufficient_n", name
    # And no line anywhere reads pass.
    assert all(line.verdict != "pass" for line in sc.lines)


def test_win_rate_and_overrides_are_diagnostic_regardless_of_n() -> None:
    """Session-level stats are never verdicted -- not below n, not above it."""
    small = metrics.scorecard(_winners(8))
    big = metrics.scorecard(_winners(120))
    for sc in (small, big):
        assert sc.get("win_rate").verdict == "diagnostic"
        assert sc.get("override_expectancy_r").verdict == "diagnostic"


def test_every_gated_metric_shares_the_expectancy_gate_as_its_required_n() -> None:
    """`its required n` (T027 spec) resolves to the M-8 expectancy gate for
    every gated metric: one below the bar → all withheld; at the bar → the
    withholding lifts (individual pass/fail then depends on the value)."""
    just_below = metrics.scorecard(_winners(metrics.MIN_N_FOR_EXPECTANCY - 1))
    for name in _GATED:
        assert just_below.get(name).verdict == "insufficient_n", name

    at_bar = metrics.scorecard(_winners(metrics.MIN_N_FOR_EXPECTANCY))
    assert all(at_bar.get(name).verdict != "insufficient_n" for name in _GATED)


def test_section3_required_n_constants_are_ordered() -> None:
    """The § 3 table's thresholds exist and ascend: plumbing (~20) <
    expectancy/M-8 gate (~100) < significance (~400)."""
    assert metrics.MIN_N_FOR_PLUMBING < metrics.MIN_N_FOR_EXPECTANCY
    assert metrics.MIN_N_FOR_EXPECTANCY < metrics.MIN_N_FOR_SIGNIFICANCE
    assert metrics.MIN_N_FOR_PLUMBING == 20
    assert metrics.MIN_N_FOR_EXPECTANCY == 100


def test_below_plumbing_n_warns_the_plumbing_isnt_even_proven() -> None:
    sc = metrics.scorecard(_winners(5))
    assert any("plumbing" in w for w in sc.warnings)


# --------------------------------------------------------------------------- #
# Render side -- the dashboard refuses to display a verdict below n
# --------------------------------------------------------------------------- #


def test_dashboard_shows_n_too_low_not_a_green_check() -> None:
    html = _digest_html(metrics.scorecard(_winners(8)))
    # Human-readable withholding, and NO green pass verdict RENDERED anywhere.
    # (Assert on the element's class attribute, not the bare token -- the CSS
    # rule `.tc-verdict--pass{...}` lives in the always-present stylesheet.)
    assert "n too low" in html
    assert "tc-verdict tc-verdict--pass" not in html
    # The section is visibly de-emphasised and the withholding is spelled out.
    assert "tc-scorecard tc-scorecard--greyed" in html
    assert 'class="tc-sc-withheld"' in html


def test_dashboard_labels_session_stats_diagnostic_not_evidence() -> None:
    # Always -- even in an "honest" (n>=bar) scorecard, win_rate stays
    # labelled, never a verdict.
    honest = metrics.scorecard(_winners(8), n_bar=8)
    html = _digest_html(honest)
    assert "diagnostic — not evidence" in html


def test_dashboard_shows_pass_only_once_the_sample_is_large_enough() -> None:
    """The mirror image: the SAME flawless numbers, but now with enough
    trades that the bar is met, DO render a green verdict. This proves the
    withholding is driven by sample size, not by a blanket suppression."""
    honest = metrics.scorecard(_winners(8), n_bar=8)
    html = _digest_html(honest)
    assert honest.honest is True
    assert "tc-verdict tc-verdict--pass" in html  # expectancy clears its bar and shows
    assert 'class="tc-sc-withheld"' not in html  # nothing withheld now
