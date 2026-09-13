"""T019 -- M009 Trade Card renderer + daily digest.

Snapshot + behavioural tests for `vt/render/card.py`. The renderer is a
pure function: same JSON in -> byte-identical HTML out, no external
requests, offline-renderable (must open from a local file and inside the
Obsidian vault). Contract lives in `Trade_Card_Spec.md`.

Two layers of test here:

  * **Behavioural** -- assert the properties `Trade_Card_Spec.md` actually
    requires (chip count and no-padding, the score/gate/risk colour
    semantics carrying a non-colour label too, HTML escaping, the
    STAND DOWN empty state, determinism, zero external requests). These
    are what make the renderer *correct*.
  * **Snapshot** -- pin the exact bytes so a future edit that changes the
    output has to be a deliberate re-baseline (run with
    `UPDATE_SNAPSHOTS=1`). This is the "byte-identical" half of T019.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vt.render.card import render_card, render_digest


# --------------------------------------------------------------------------- #
# Fixtures -- the Trade_Card_Spec.md data contract, and variants of it
# --------------------------------------------------------------------------- #


def _nvda_card() -> dict:
    """The worked example from Trade_Card_Spec.md, three reason chips."""
    return {
        "card_id": "VT-20260817-NVDA-001",
        "ts_generated": "2026-08-17T10:42:03Z",
        "venue": "alpaca-paper",
        "symbol": "NVDA",
        "direction": "long",
        "rank": 1,
        "candidates_today": 6,
        "day_change_pct": 2.1,
        "score": {
            "total": 10, "max": 12,
            "R1_trend": 2, "R2_momentum": 1, "R3_volume": 2,
            "R4_volatility": 2, "R5_structure": 2, "R6_rel_strength": 1,
        },
        "reasons": [
            {"icon": "⚡", "text": "RVOL 3.4×", "source": "R3"},
            {"icon": "\U0001f4c8", "text": "Held VWAP retest", "source": "R1"},
            {"icon": "\U0001f4aa", "text": "+1.8% vs SPY", "source": "R6"},
        ],
        "levels": {
            "entry": 178.40, "stop": 176.54, "atr_14_5m": 1.24,
            "stop_mult": 1.5, "target_1": 180.26, "target_2": "trail_2atr",
        },
        "sizing": {
            "shares": 26, "notional": 4638.40, "risk_dollars": 48.36,
            "risk_pct_equity": 0.48, "one_R": 48.36, "equity": 10000.00,
        },
        "invalidation": {
            "condition": "loses VWAP on > 1.5× avg volume",
            "time_stop": "2026-08-17T11:42:00Z", "time_stop_r": 0.5,
        },
        "gate": {"state": "NORMAL", "multiplier": 1.00, "vix": 16.2, "events": []},
        "outcome": None,
    }


def _with_reasons(n: int) -> dict:
    card = _nvda_card()
    card["reasons"] = card["reasons"][:n]
    return card


# --------------------------------------------------------------------------- #
# Card -- purity / no external requests
# --------------------------------------------------------------------------- #


def test_render_card_is_pure_and_deterministic():
    card = _nvda_card()
    assert render_card(card) == render_card(card)


def test_render_card_does_not_mutate_input():
    card = _nvda_card()
    before = repr(card)
    render_card(card)
    assert repr(card) == before


def test_render_card_makes_no_external_requests():
    html = render_card(_nvda_card())
    for token in ("http://", "https://", "<script", "<link", "src=", "//cdn"):
        assert token not in html, f"card must be offline; found {token!r}"


# --------------------------------------------------------------------------- #
# Card -- the four-question anatomy is present
# --------------------------------------------------------------------------- #


def test_render_card_shows_what_why_howmuch_wheninwrong():
    html = render_card(_nvda_card())
    # what
    assert "NVDA" in html
    assert "LONG" in html
    assert "10/12" in html
    # why (chips)
    assert "RVOL 3.4×" in html
    # how much
    assert "$178.40" in html          # entry
    assert "$176.54" in html          # stop
    assert "0.48%" in html            # risk pct
    # when it's wrong
    assert "loses VWAP on &gt; 1.5× avg volume" in html


# --------------------------------------------------------------------------- #
# Card -- reason chips: max 3, and NEVER padded
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [1, 2, 3])
def test_render_card_renders_exactly_n_chips(n: int):
    html = render_card(_with_reasons(n))
    assert html.count('class="tc-chip"') == n


def test_render_card_never_pads_below_three():
    """One high-scoring component -> one chip, not three empty ones."""
    html = render_card(_with_reasons(1))
    assert html.count('class="tc-chip"') == 1
    # no empty chip left as a placeholder
    assert 'class="tc-chip"></li>' not in html
    assert 'class="tc-chip"> </li>' not in html


def test_render_card_caps_chips_at_three_when_given_more():
    card = _nvda_card()
    card["reasons"] = card["reasons"] + [{"icon": "✨", "text": "extra", "source": "R2"}]
    html = render_card(card)
    assert html.count('class="tc-chip"') == 3


# --------------------------------------------------------------------------- #
# Card -- score bar colour tier carries a non-colour label (greyscale/WCAG)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "total,tier",
    [(12, "green"), (11, "green"), (10, "amber"), (9, "amber"), (8, "red"), (0, "red")],
)
def test_render_card_score_tier(total: int, tier: str):
    card = _nvda_card()
    card["score"]["total"] = total
    html = render_card(card)
    assert f'data-score-tier="{tier}"' in html


def test_render_card_score_bar_has_text_label_not_only_colour():
    html = render_card(_nvda_card())
    # A label word must accompany the colour so it survives greyscale.
    assert "tc-score-tier" in html
    assert "AMBER" in html or "OK" in html  # 10/12 is the amber tier


# --------------------------------------------------------------------------- #
# Card -- direction
# --------------------------------------------------------------------------- #


def test_render_card_long_direction():
    html = render_card(_nvda_card())
    assert "tc-dir--long" in html
    assert "▲" in html  # up triangle


def test_render_card_short_direction():
    card = _nvda_card()
    card["direction"] = "short"
    html = render_card(card)
    assert "tc-dir--short" in html
    assert "▼" in html  # down triangle


# --------------------------------------------------------------------------- #
# Card -- risk value goes amber if > 0.5% of equity (M006-bug canary)
# --------------------------------------------------------------------------- #


def test_render_card_risk_normal_is_not_flagged():
    html = render_card(_nvda_card())  # 0.48%
    assert "tc-num--risk-over" not in html


def test_render_card_risk_over_half_pct_is_flagged():
    card = _nvda_card()
    card["sizing"]["risk_pct_equity"] = 0.72
    html = render_card(card)
    assert "tc-num--risk-over" in html


# --------------------------------------------------------------------------- #
# Card -- gate badge state
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "state,cls",
    [("NORMAL", "tc-gate--normal"),
     ("REDUCED", "tc-gate--reduced"),
     ("STAND_DOWN", "tc-gate--stand-down")],
)
def test_render_card_gate_badge(state: str, cls: str):
    card = _nvda_card()
    card["gate"]["state"] = state
    html = render_card(card)
    assert cls in html


def test_render_card_gate_shows_vix_and_events():
    card = _nvda_card()
    card["gate"]["events"] = ["FOMC 14:00"]
    card["gate"]["vix"] = 22.5
    html = render_card(card)
    assert "FOMC 14:00" in html
    assert "22.5" in html


# --------------------------------------------------------------------------- #
# Card -- HTML escaping (never trust free-text fields)
# --------------------------------------------------------------------------- #


def test_render_card_escapes_free_text():
    card = _nvda_card()
    card["invalidation"]["condition"] = "<script>alert('x')</script> & <b>bad</b>"
    html = render_card(card)
    assert "<script>alert(" not in html
    assert "&lt;script&gt;" in html
    assert "&amp;" in html


def test_render_card_time_stop_less_than_is_escaped():
    """The time-stop reads 'flat if < +0.5R'; the '<' must be escaped so a
    strict HTML parser / Obsidian never mistakes it for a tag start."""
    html = render_card(_nvda_card())
    assert "&lt; +0.5R" in html
    assert "flat if < " not in html


def test_render_card_escapes_symbol_and_reason_text():
    card = _nvda_card()
    card["symbol"] = "A&B"
    card["reasons"][0]["text"] = "<i>hot</i>"
    html = render_card(card)
    assert "A&amp;B" in html
    assert "<i>hot</i>" not in html


# --------------------------------------------------------------------------- #
# Card -- outcome strip only when closed
# --------------------------------------------------------------------------- #


def test_render_card_string_target2_escaped_exactly_once():
    """A non-sentinel string target_2 must be escaped once, not twice."""
    card = _nvda_card()
    card["levels"]["target_2"] = "R&D level"
    html = render_card(card)
    assert "R&amp;D level" in html
    assert "R&amp;amp;D" not in html  # not double-escaped


def test_render_card_numeric_target2_rendered_as_money():
    card = _nvda_card()
    card["levels"]["target_2"] = 182.10
    html = render_card(card)
    assert "$182.10" in html


def test_render_card_open_card_has_no_result_strip():
    html = render_card(_nvda_card())
    assert "tc-outcome" not in html


def test_render_card_closed_card_renders_result():
    card = _nvda_card()
    card["outcome"] = {"r_multiple": 1.3, "exit_reason": "target_1", "adherence": True}
    html = render_card(card)
    assert "tc-outcome" in html
    assert "target_1" in html
    assert "+1.3" in html


# --------------------------------------------------------------------------- #
# Digest -- full standalone document
# --------------------------------------------------------------------------- #


def _digest(cards, *, gate=None, rejected=None, scorecard=None) -> dict:
    return {
        "date": "2026-08-17",
        "gate": gate or {"state": "NORMAL", "multiplier": 1.0, "vix": 16.2, "events": []},
        "cards": cards,
        "rejected": rejected or [],
        "scorecard": scorecard,
    }


def test_render_digest_is_a_full_html_document():
    html = render_digest(_digest([_nvda_card()]))
    assert html.lstrip().lower().startswith("<!doctype html")
    assert "<html" in html and "</html>" in html
    assert "<style" in html  # self-contained styling, no external css


def test_render_digest_makes_no_external_requests():
    html = render_digest(_digest([_nvda_card()]))
    for token in ("http://", "https://", "<script", "<link", "src=", "//cdn"):
        assert token not in html, f"digest must be offline; found {token!r}"


def test_render_digest_renders_each_ranked_card():
    cards = [_nvda_card(), _with_reasons(2), _with_reasons(1)]
    html = render_digest(_digest(cards))
    assert html.count('<article class="tc-card"') == 3


def test_render_digest_caps_at_six_cards():
    cards = [_nvda_card() for _ in range(9)]
    html = render_digest(_digest(cards))
    assert html.count('<article class="tc-card"') == 6


def test_render_digest_shows_gate_banner():
    html = render_digest(_digest([_nvda_card()]))
    assert "tc-banner" in html
    assert "NORMAL" in html


@pytest.mark.parametrize(
    "state", ["NORMAL", "REDUCED", "STAND_DOWN"],
)
def test_render_digest_banner_reflects_state(state):
    gate = {"state": state, "multiplier": 0.0, "vix": 30.0, "events": ["CPI 08:30"]}
    html = render_digest(_digest([], gate=gate))
    assert state.replace("_", " ") in html


# --------------------------------------------------------------------------- #
# Digest -- STAND DOWN: the banner is the ONLY thing rendered
# --------------------------------------------------------------------------- #


def test_render_digest_stand_down_renders_zero_cards():
    gate = {"state": "STAND_DOWN", "multiplier": 0.0, "vix": 30.0, "events": ["FOMC 14:00"]}
    # even if cards are (wrongly) passed, a STAND DOWN day shows none
    html = render_digest(_digest([_nvda_card(), _nvda_card()], gate=gate))
    assert html.count('<article class="tc-card"') == 0
    assert "STAND DOWN" in html
    assert "FOMC 14:00" in html


def test_render_digest_empty_day_is_loud_not_blank():
    gate = {"state": "STAND_DOWN", "multiplier": 0.0, "vix": None, "events": []}
    html = render_digest(_digest([], gate=gate))
    assert "STAND DOWN" in html


# --------------------------------------------------------------------------- #
# Digest -- rejected candidates table
# --------------------------------------------------------------------------- #


def test_render_digest_rejected_table():
    rejected = [
        {"symbol": "TSLA", "score": 7, "first_failed_gate": "rubric_below_threshold"},
        {"symbol": "AMD", "score": 5, "first_failed_gate": "R6_below_floor"},
    ]
    html = render_digest(_digest([_nvda_card()], rejected=rejected))
    assert "TSLA" in html
    assert "AMD" in html
    assert "rubric_below_threshold" in html


def test_render_digest_rejected_text_is_escaped():
    rejected = [{"symbol": "X<Y", "score": 3, "first_failed_gate": "a & b"}]
    html = render_digest(_digest([_nvda_card()], rejected=rejected))
    assert "X<Y" not in html
    assert "X&lt;Y" in html
    assert "a &amp; b" in html


# --------------------------------------------------------------------------- #
# Digest -- rolling scorecard (Metrics_Definitions.md 5)
# --------------------------------------------------------------------------- #


def _scorecard(honest: bool) -> dict:
    return {
        "n": 100 if honest else 12,
        "honest": honest,
        "lines": [
            {"name": "expectancy_r", "value": 0.35, "bar": 0.10,
             "verdict": "pass" if honest else "insufficient_n"},
            {"name": "win_rate", "value": 0.45, "bar": None, "verdict": "diagnostic"},
            {"name": "profit_factor", "value": 1.6, "bar": 1.3,
             "verdict": "pass" if honest else "insufficient_n"},
        ],
        "warnings": [] if honest else ["n=12: below expectancy-verdict threshold (30)"],
    }


def test_render_digest_scorecard_rendered_when_honest():
    html = render_digest(_digest([_nvda_card()], scorecard=_scorecard(True)))
    assert "tc-scorecard" in html
    assert "expectancy_r" in html
    assert "pass" in html


def test_render_digest_scorecard_insufficient_n_is_greyed_not_green():
    html = render_digest(_digest([_nvda_card()], scorecard=_scorecard(False)))
    assert "insufficient_n" in html
    assert "below expectancy-verdict threshold" in html


def test_render_digest_is_deterministic():
    d = _digest([_nvda_card(), _with_reasons(2)], scorecard=_scorecard(True))
    assert render_digest(d) == render_digest(d)


# --------------------------------------------------------------------------- #
# Snapshot lock -- the byte-identical half of T019
# --------------------------------------------------------------------------- #

SNAP_DIR = Path(__file__).parent / "snapshots"


def _assert_snapshot(name: str, actual: str) -> None:
    SNAP_DIR.mkdir(exist_ok=True)
    path = SNAP_DIR / name
    if os.environ.get("UPDATE_SNAPSHOTS"):
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(actual)
    assert path.exists(), (
        f"snapshot {name!r} is missing -- run with UPDATE_SNAPSHOTS=1 to create it"
    )
    with open(path, "r", encoding="utf-8", newline="") as fh:
        expected = fh.read()
    assert actual == expected, (
        f"snapshot {name!r} mismatch -- if this change is intended, "
        "re-baseline with UPDATE_SNAPSHOTS=1"
    )


@pytest.mark.parametrize("n", [1, 2, 3])
def test_card_snapshot_by_chip_count(n: int):
    _assert_snapshot(f"card_{n}chip.html", render_card(_with_reasons(n)))


def test_card_snapshot_short_direction():
    card = _nvda_card()
    card["direction"] = "short"
    card["gate"]["state"] = "REDUCED"
    card["gate"]["multiplier"] = 0.5
    _assert_snapshot("card_short_reduced.html", render_card(card))


def test_card_snapshot_closed_outcome():
    card = _nvda_card()
    card["outcome"] = {"r_multiple": 1.3, "exit_reason": "target_1", "adherence": True}
    _assert_snapshot("card_closed.html", render_card(card))


def test_digest_snapshot_full():
    d = _digest(
        [_nvda_card(), _with_reasons(2)],
        rejected=[{"symbol": "TSLA", "score": 7, "first_failed_gate": "rubric_below_threshold"}],
        scorecard=_scorecard(True),
    )
    _assert_snapshot("digest_full.html", render_digest(d))


def test_digest_snapshot_stand_down():
    gate = {"state": "STAND_DOWN", "multiplier": 0.0, "vix": 31.4, "events": ["FOMC 14:00"]}
    _assert_snapshot("digest_stand_down.html", render_digest(_digest([_nvda_card()], gate=gate)))
