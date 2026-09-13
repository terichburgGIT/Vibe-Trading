"""M009 -- Trade Card renderer + daily digest. Phase D, T019.

Pure JSON -> HTML per `Trade_Card_Spec.md`. Two entry points:

  * ``render_card(card) -> str``   -- one Trade Card as a self-contained
    ``<article>`` fragment (the digest embeds many of these).
  * ``render_digest(digest) -> str`` -- the full standalone daily digest
    document: gate banner, ranked cards, rejected table, rolling
    scorecard.

Three properties this module guarantees, all of which T019 checks:

  * **Pure.** Same JSON in -> byte-identical HTML out. No ``datetime.now``,
    no randomness, no dict-order dependence (fixed component keys, input
    list order preserved). This is what lets a card be regenerated from
    the journal at any time and still match -- ``Trade_Card_Spec.md``:
    "Render is pure: same JSON in -> same HTML out."
  * **Offline.** No external requests: CSS is inlined, icons are emoji
    (text, not images), there is no ``<script>``, ``<link>``, or remote
    ``src``. Must open from a local file and from inside the Obsidian
    vault.
  * **Safe.** Every free-text field (symbol, reason text, invalidation
    condition, event labels, exit reason, rejection reasons) is
    HTML-escaped. The card is written from data that partly originates
    outside our control (symbols, LLM-authored thesis text), so the
    renderer never trusts it.

Design note on ``render_digest``'s signature. ``03_Modules.md`` nominally
lists ``render_digest(date)``. A *pure* renderer cannot read the journal
itself (that would be I/O and would break byte-identity), so this takes
the already-assembled digest mapping -- ``{date, gate, cards, rejected,
scorecard}`` -- and the caller (the pipeline / journal layer) is
responsible for gathering it. The ``date`` lives inside that mapping.
Wiring a ``build_digest(date)`` convenience that reads the journal is
deferred until the written-card shape is enriched to the full
``Trade_Card_Spec.md`` data contract (the pipeline currently journals a
leaner card -- see ``vt/pipeline/runner.py`` ``_write_card``).

Full contract in `03_Modules.md` section M009; tests in
`vt/tests/test_render_card.py`.
"""

from __future__ import annotations

import html
from typing import Any, Mapping, Sequence


# --------------------------------------------------------------------------- #
# Constants -- Trade_Card_Spec.md
# --------------------------------------------------------------------------- #

MAX_REASON_CHIPS = 3          # "Max 3 ... never pad"
MAX_DIGEST_CARDS = 6          # "Ranked cards -- highest score first, max 6"
RISK_PCT_LIMIT = 0.5          # amber canary: risk > 0.5% of equity => M006 bug
SCORE_MAX_DEFAULT = 12

# Score-bar colour tiers: red < 9, amber 9-10, green 11-12.
_TIER_GREEN_MIN = 11
_TIER_AMBER_MIN = 9
_TIER_LABEL = {"green": "STRONG", "amber": "OK", "red": "BELOW"}


# --------------------------------------------------------------------------- #
# Formatting helpers -- all deterministic, all pure
# --------------------------------------------------------------------------- #


def _esc(value: Any) -> str:
    """HTML-escape any value. ``None`` renders as an em dash."""
    if value is None:
        return "—"
    return html.escape(str(value), quote=True)


def _money(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return "$" + f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return _esc(value)


def _pct(value: Any, *, dp: int = 2, signed: bool = False) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return _esc(value)
    body = f"{v:+.{dp}f}" if signed else f"{v:.{dp}f}"
    return body + "%"


def _qty(value: Any) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return _esc(value)
    if v == int(v):
        return str(int(v))
    return f"{v:.8f}".rstrip("0").rstrip(".")


def _r(value: Any, *, signed: bool = True) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return _esc(value)
    body = f"{v:+.1f}" if signed else f"{v:.1f}"
    return body + "R"


def _hhmm(iso: Any) -> str | None:
    """HH:MM from an ISO-8601 timestamp string, as-given (no tz maths --
    that would need external tz data and break offline purity)."""
    if not isinstance(iso, str) or "T" not in iso:
        return None
    return iso.split("T", 1)[1][:5]


def _score_tier(total: int) -> str:
    if total >= _TIER_GREEN_MIN:
        return "green"
    if total >= _TIER_AMBER_MIN:
        return "amber"
    return "red"


def _target_r(entry: Any, stop: Any, target: Any, direction: str) -> str | None:
    """R-multiple of a target relative to the entry/stop risk unit.
    Returns None when it can't be computed cleanly (missing/zero risk)."""
    try:
        e, s, t = float(entry), float(stop), float(target)
    except (TypeError, ValueError):
        return None
    risk_per_unit = (e - s) if direction != "short" else (s - e)
    if risk_per_unit <= 0:
        return None
    reward = (t - e) if direction != "short" else (e - t)
    return _r(reward / risk_per_unit)


# --------------------------------------------------------------------------- #
# Card fragment
# --------------------------------------------------------------------------- #


def render_card(card: Mapping[str, Any]) -> str:
    """Render one Trade Card to a self-contained ``<article>`` fragment.
    Pure: same mapping in -> identical string out. The input is never
    mutated. See module docstring for the guarantees."""
    symbol = card.get("symbol")
    direction = str(card.get("direction", "long")).lower()
    is_short = direction == "short"
    dir_class = "tc-dir--short" if is_short else "tc-dir--long"
    dir_arrow = "▼" if is_short else "▲"
    dir_label = "SHORT" if is_short else "LONG"

    score = card.get("score") or {}
    total = int(score.get("total", 0))
    score_max = int(score.get("max", SCORE_MAX_DEFAULT))
    tier = _score_tier(total)
    tier_label = _TIER_LABEL[tier]

    levels = card.get("levels") or {}
    sizing = card.get("sizing") or {}
    invalidation = card.get("invalidation") or {}
    gate = card.get("gate") or {}

    entry = levels.get("entry")
    stop = levels.get("stop")

    lines: list[str] = []
    lines.append(f'<article class="tc-card" data-score-tier="{tier}">')

    # ---- Header: what + conviction --------------------------------------- #
    price_bits = [_money(entry)]
    day_change = card.get("day_change_pct")
    if day_change is not None:
        price_bits.append(f"{_pct(day_change, dp=1, signed=True)} today")
    price_line = " · ".join(price_bits)

    rank = card.get("rank")
    of = card.get("candidates_today")
    rank_line = f"#{_esc(rank)} of {_esc(of)}" if rank is not None and of is not None else ""

    lines.append('  <header class="tc-header">')
    lines.append(
        f'    <div class="tc-dir {dir_class}">'
        f'<span class="tc-arrow" aria-hidden="true">{dir_arrow}</span> {dir_label}</div>'
    )
    lines.append(f'    <div class="tc-symbol">{_esc(symbol)}</div>')
    lines.append(f'    <div class="tc-price">{price_line}</div>')
    lines.append(_render_score(total, score_max, tier, tier_label))
    if rank_line:
        lines.append(f'    <div class="tc-rank">RANK {rank_line}</div>')
    lines.append("  </header>")

    # ---- Why: reason chips (max 3, never padded) ------------------------- #
    chips = _render_reason_chips(card.get("reasons") or [])
    if chips:
        lines.append(f'  <ul class="tc-reasons">{chips}</ul>')

    # ---- How much: numbers ----------------------------------------------- #
    lines.append(_render_numbers(card, entry, stop, direction, levels, sizing))

    # ---- When it's wrong: invalidation ----------------------------------- #
    inv = _render_invalidation(invalidation)
    if inv:
        lines.append(inv)

    # ---- Context: gate --------------------------------------------------- #
    lines.append(_render_gate_footer(gate))

    # ---- Result strip (only once the trade has closed) ------------------- #
    outcome = card.get("outcome")
    if outcome:
        lines.append(_render_outcome(outcome))

    lines.append("</article>")
    return "\n".join(lines)


def _render_score(total: int, score_max: int, tier: str, tier_label: str) -> str:
    filled = max(0, min(total, score_max))
    segs = []
    for i in range(score_max):
        state = "on" if i < filled else "off"
        segs.append(f'<span class="tc-seg tc-seg--{state}"></span>')
    bar = "".join(segs)
    aria = f"score {total} of {score_max}, {tier_label.lower()}"
    return (
        '    <div class="tc-score">'
        '<span class="tc-score-label">SCORE</span>'
        f'<span class="tc-score-num">{total}/{score_max}</span>'
        f'<span class="tc-score-bar" role="img" aria-label="{_esc(aria)}">{bar}</span>'
        f'<span class="tc-score-tier">{tier_label}</span>'
        "</div>"
    )


def _render_reason_chips(reasons: Sequence[Mapping[str, Any]]) -> str:
    items = []
    for r in list(reasons)[:MAX_REASON_CHIPS]:
        icon = _esc(r.get("icon", ""))
        text = _esc(r.get("text", ""))
        items.append(
            f'<li class="tc-chip">'
            f'<span class="tc-chip-icon" aria-hidden="true">{icon}</span> {text}</li>'
        )
    return "".join(items)


def _render_numbers(
    card: Mapping[str, Any],
    entry: Any,
    stop: Any,
    direction: str,
    levels: Mapping[str, Any],
    sizing: Mapping[str, Any],
) -> str:
    cells: list[str] = []

    def cell(label: str, value: str, *, extra: str = "") -> str:
        cls = "tc-num" + (f" {extra}" if extra else "")
        return (
            f'<div class="{cls}">'
            f'<span class="tc-num-label">{label}</span>'
            f'<span class="tc-num-value">{value}</span></div>'
        )

    # ENTRY
    cells.append(cell("ENTRY", _money(entry)))

    # STOP -- always red-tinted; show the stop distance as a % of entry.
    stop_extra = ""
    stop_mult = levels.get("stop_mult")
    if entry is not None and stop is not None:
        try:
            dist_pct = (float(stop) - float(entry)) / float(entry) * 100.0
            mult_txt = f"{_qty(stop_mult)}×ATR " if stop_mult is not None else ""
            stop_extra = f' <span class="tc-num-sub">{mult_txt}{_pct(dist_pct, signed=True)}</span>'
        except (TypeError, ValueError, ZeroDivisionError):
            stop_extra = ""
    cells.append(cell("STOP", _money(stop) + stop_extra, extra="tc-num--stop"))

    # SIZE -- "sh" is the equities unit noun from the worked example; a
    # crypto-aware unit label is a Designer-pass refinement (M-6), not a
    # functional-render concern (Trade_Card_Spec.md implementation notes).
    shares = sizing.get("shares")
    notional = sizing.get("notional")
    cells.append(cell("SIZE", f"{_qty(shares)} sh · {_money(notional)}"))

    # RISK -- amber canary if > 0.5% of equity (should be impossible; if it
    # renders amber, M006 has a sizing bug).
    risk_pct = sizing.get("risk_pct_equity")
    risk_over = False
    try:
        risk_over = risk_pct is not None and float(risk_pct) > RISK_PCT_LIMIT
    except (TypeError, ValueError):
        risk_over = False
    risk_extra = "tc-num--risk-over" if risk_over else "tc-num--risk"
    risk_val = f"{_money(sizing.get('risk_dollars'))} · 1R · {_pct(risk_pct)}"
    cells.append(cell("RISK", risk_val, extra=risk_extra))

    # TARGET 1
    t1 = levels.get("target_1")
    if t1 is not None:
        t1r = _target_r(entry, stop, t1, direction)
        t1_val = _money(t1) + (f' <span class="tc-num-sub">{t1r}</span>' if t1r else "")
        cells.append(cell("TARGET 1", t1_val))

    # TARGET 2 -- a trailing-stop sentinel, another string label, or a price.
    t2 = levels.get("target_2")
    if t2 is not None:
        if t2 == "trail_2atr":
            t2_val = "trail 2×ATR"
        elif isinstance(t2, str):
            t2_val = _esc(t2)  # arbitrary label -- escape exactly once
        else:
            t2_val = _money(t2)
        cells.append(cell("TARGET 2", t2_val))

    return '  <section class="tc-numbers">' + "".join(cells) + "</section>"


def _render_invalidation(inv: Mapping[str, Any]) -> str:
    condition = inv.get("condition")
    time_stop = _hhmm(inv.get("time_stop"))
    time_stop_r = inv.get("time_stop_r")
    if condition is None and time_stop is None:
        return ""
    rows = []
    if condition is not None:
        rows.append(
            f'<div class="tc-invalid">'
            f'<span class="tc-invalid-icon" aria-hidden="true">✕</span> '
            f'<span class="tc-invalid-label">INVALID IF</span> {_esc(condition)}</div>'
        )
    if time_stop is not None:
        r_txt = f"&lt; {_r(time_stop_r)} " if time_stop_r is not None else ""
        rows.append(
            f'<div class="tc-timestop">'
            f'<span class="tc-invalid-icon" aria-hidden="true">⏱</span> '
            f'<span class="tc-invalid-label">TIME STOP</span> flat if {r_txt}by {_esc(time_stop)}</div>'
        )
    return '  <section class="tc-invalidation">' + "".join(rows) + "</section>"


def _render_gate_footer(gate: Mapping[str, Any]) -> str:
    state = str(gate.get("state", "NORMAL"))
    state_class = "tc-gate--" + state.lower().replace("_", "-")
    state_label = state.replace("_", " ")

    segments = [f"{state_label}"]
    mult = gate.get("multiplier")
    if mult is not None:
        try:
            segments[0] += f" ({float(mult):.2f}×)"
        except (TypeError, ValueError):
            pass

    events = gate.get("events") or []
    if events:
        segments.append(", ".join(_esc(e) for e in events))
    else:
        segments.append("no events")

    vix = gate.get("vix")
    if vix is not None:
        try:
            segments.append(f"VIX {float(vix):.1f}")
        except (TypeError, ValueError):
            segments.append(f"VIX {_esc(vix)}")

    body = " · ".join(segments)
    return (
        f'  <footer class="tc-gate {state_class}">'
        f'<span class="tc-gate-label">GATE</span> {body}</footer>'
    )


def _render_outcome(outcome: Mapping[str, Any]) -> str:
    r = outcome.get("r_multiple")
    win = False
    try:
        win = r is not None and float(r) >= 0
    except (TypeError, ValueError):
        win = False
    cls = "tc-outcome--win" if win else "tc-outcome--loss"
    adherence = outcome.get("adherence")
    if adherence is True:
        adh = "adhered"
    elif adherence is False:
        adh = "OVERRODE PLAN"
    else:
        adh = None
    bits = [f"RESULT {_r(r)}", _esc(outcome.get("exit_reason"))]
    if adh is not None:
        bits.append(adh)
    return f'  <footer class="tc-outcome {cls}">' + " · ".join(bits) + "</footer>"


# --------------------------------------------------------------------------- #
# Digest document
# --------------------------------------------------------------------------- #


def render_digest(digest: Mapping[str, Any]) -> str:
    """Render the full daily digest as a standalone HTML document.

    On a STAND DOWN day the gate banner is the only thing rendered -- no
    cards, no rejected table, no scorecard. Not trading is a decision the
    system made, and the empty state is deliberately loud
    (`Trade_Card_Spec.md`, "The daily digest").
    """
    date = digest.get("date")
    gate = digest.get("gate") or {}
    state = str(gate.get("state", "NORMAL"))
    stand_down = state == "STAND_DOWN"

    body: list[str] = ['<main class="tc-digest">', _render_banner(date, gate)]

    if stand_down:
        body.append(
            '  <section class="tc-standdown">'
            "<p>STAND DOWN — no trades today. Not trading is the decision.</p>"
            "</section>"
        )
    else:
        cards = list(digest.get("cards") or [])[:MAX_DIGEST_CARDS]
        if cards:
            body.append('  <section class="tc-cards">')
            body.extend(render_card(c) for c in cards)
            body.append("  </section>")
        else:
            body.append(
                '  <section class="tc-cards tc-cards--empty">'
                "<p>No candidates cleared the screen today.</p></section>"
            )
        rejected = _render_rejected(digest.get("rejected") or [])
        if rejected:
            body.append(rejected)
        scorecard = _render_scorecard(digest.get("scorecard"))
        if scorecard:
            body.append(scorecard)

    body.append("</main>")

    title = f"VibeTrading — Daily Digest {_esc(date)}"
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{title}</title>",
            f"<style>{_STYLESHEET}</style>",
            "</head>",
            "<body>",
            "\n".join(body),
            "</body>",
            "</html>",
            "",
        ]
    )


def _render_banner(date: Any, gate: Mapping[str, Any]) -> str:
    state = str(gate.get("state", "NORMAL"))
    state_class = "tc-banner--" + state.lower().replace("_", "-")
    state_label = state.replace("_", " ")

    detail_bits = []
    mult = gate.get("multiplier")
    if mult is not None:
        try:
            detail_bits.append(f"{float(mult):.2f}× size")
        except (TypeError, ValueError):
            pass
    vix = gate.get("vix")
    if vix is not None:
        try:
            detail_bits.append(f"VIX {float(vix):.1f}")
        except (TypeError, ValueError):
            detail_bits.append(f"VIX {_esc(vix)}")
    events = gate.get("events") or []
    detail_bits.append(", ".join(_esc(e) for e in events) if events else "no events")

    return (
        f'  <header class="tc-banner {state_class}">'
        f'<div class="tc-banner-date">{_esc(date)}</div>'
        f'<div class="tc-banner-state">{state_label}</div>'
        f'<div class="tc-banner-detail">{" · ".join(detail_bits)}</div>'
        "</header>"
    )


def _render_rejected(rejected: Sequence[Mapping[str, Any]]) -> str:
    if not rejected:
        return ""
    rows = []
    for r in rejected:
        rows.append(
            "<tr>"
            f'<td class="tc-rej-sym">{_esc(r.get("symbol"))}</td>'
            f'<td class="tc-rej-score">{_esc(r.get("score"))}</td>'
            f'<td class="tc-rej-gate">{_esc(r.get("first_failed_gate"))}</td>'
            "</tr>"
        )
    return (
        '  <section class="tc-rejected">'
        "<h2>Rejected candidates</h2>"
        '<table class="tc-rej-table">'
        "<thead><tr><th>Symbol</th><th>Score</th><th>First failed gate</th></tr></thead>"
        f'<tbody>{"".join(rows)}</tbody>'
        "</table></section>"
    )


def _render_scorecard(sc: Mapping[str, Any] | None) -> str:
    if not sc:
        return ""
    n = sc.get("n", 0)
    honest = bool(sc.get("honest", False))
    honest_class = "" if honest else " tc-scorecard--greyed"

    rows = []
    for line in sc.get("lines") or []:
        verdict = str(line.get("verdict", "diagnostic"))
        bar = line.get("bar")
        bar_txt = "—" if bar is None else f"{float(bar):.2f}"
        rows.append(
            "<tr>"
            f'<td class="tc-sc-name">{_esc(line.get("name"))}</td>'
            f'<td class="tc-sc-value">{_esc(_fmt_metric(line.get("value")))}</td>'
            f'<td class="tc-sc-bar">{bar_txt}</td>'
            f'<td class="tc-verdict tc-verdict--{verdict}">{verdict}</td>'
            "</tr>"
        )

    warnings = sc.get("warnings") or []
    warn_html = ""
    if warnings:
        items = "".join(f"<li>{_esc(w)}</li>" for w in warnings)
        warn_html = f'<ul class="tc-sc-warnings">{items}</ul>'

    return (
        f'  <section class="tc-scorecard{honest_class}">'
        f"<h2>Rolling scorecard (n={_esc(n)})</h2>"
        '<table class="tc-sc-table">'
        "<thead><tr><th>Metric</th><th>Value</th><th>Bar</th><th>Verdict</th></tr></thead>"
        f'<tbody>{"".join(rows)}</tbody>'
        "</table>"
        f"{warn_html}"
        "</section>"
    )


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


# --------------------------------------------------------------------------- #
# Stylesheet -- inlined, no external references (offline + vault-safe).
# Colour is never the only channel: score tier, gate badge and verdict all
# carry a text label too, and the sheet must survive greyscale + WCAG AA.
# --------------------------------------------------------------------------- #

_STYLESHEET = """
:root{
  --bg:#f6f7f9; --surface:#ffffff; --ink:#161a1d; --muted:#5b6570;
  --line:#e2e6ea; --green:#137a4b; --amber:#8a5a00; --red:#b3261e;
  --green-bg:#e7f4ec; --amber-bg:#fbf1dc; --red-bg:#fbe6e5;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#14171a; --surface:#1d2226; --ink:#e8ecef; --muted:#9aa4ad;
    --line:#2b3238; --green:#5cd39a; --amber:#e0b25a; --red:#f0857c;
    --green-bg:#12271d; --amber-bg:#2a2313; --red-bg:#2b1614;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.tc-digest{max-width:860px;margin:0 auto;padding:24px 16px 64px}
h2{font-size:.85rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
  margin:32px 0 10px}

/* Gate banner */
.tc-banner{display:flex;align-items:baseline;gap:16px;flex-wrap:wrap;
  padding:16px 20px;border-radius:12px;border:2px solid var(--line);margin-bottom:8px}
.tc-banner-date{font-variant-numeric:tabular-nums;color:var(--muted)}
.tc-banner-state{font-weight:700;letter-spacing:.04em}
.tc-banner-detail{color:var(--muted);font-size:.9rem}
.tc-banner--normal{border-color:var(--green);background:var(--green-bg)}
.tc-banner--reduced{border-color:var(--amber);background:var(--amber-bg)}
.tc-banner--stand-down{border-color:var(--red);background:var(--red-bg)}

.tc-standdown{padding:40px 20px;text-align:center;font-size:1.1rem;color:var(--red);
  border:2px dashed var(--red);border-radius:12px;margin-top:12px;font-weight:600}
.tc-cards--empty p,.tc-standdown p{margin:0}
.tc-cards--empty{padding:28px 20px;text-align:center;color:var(--muted)}

/* Card */
.tc-cards{display:grid;gap:14px;margin-top:8px}
.tc-card{background:var(--surface);border:1px solid var(--line);border-radius:12px;
  padding:14px 16px;box-shadow:0 1px 2px rgba(0,0,0,.05)}
.tc-header{display:grid;grid-template-columns:auto 1fr auto;align-items:center;
  gap:6px 12px;padding-bottom:10px;border-bottom:1px solid var(--line)}
.tc-dir{font-weight:700;letter-spacing:.05em}
.tc-dir--long{color:var(--green)}
.tc-dir--short{color:var(--red)}
.tc-symbol{font-size:1.35rem;font-weight:700}
.tc-price{color:var(--muted);font-variant-numeric:tabular-nums}
.tc-score{grid-column:1/-1;display:flex;align-items:center;gap:8px;font-size:.82rem}
.tc-score-label{color:var(--muted);letter-spacing:.06em}
.tc-score-num{font-weight:700;font-variant-numeric:tabular-nums}
.tc-score-bar{display:inline-flex;gap:2px}
.tc-seg{width:12px;height:10px;border-radius:2px;background:var(--line)}
.tc-card[data-score-tier=green] .tc-seg--on{background:var(--green)}
.tc-card[data-score-tier=amber] .tc-seg--on{background:var(--amber)}
.tc-card[data-score-tier=red] .tc-seg--on{background:var(--red)}
.tc-score-tier{font-weight:700;letter-spacing:.05em}
.tc-card[data-score-tier=green] .tc-score-tier{color:var(--green)}
.tc-card[data-score-tier=amber] .tc-score-tier{color:var(--amber)}
.tc-card[data-score-tier=red] .tc-score-tier{color:var(--red)}
.tc-rank{color:var(--muted);font-size:.8rem;text-align:right}

/* Reason chips */
.tc-reasons{list-style:none;display:flex;flex-wrap:wrap;gap:8px;margin:12px 0;padding:0}
.tc-chip{background:var(--bg);border:1px solid var(--line);border-radius:999px;
  padding:4px 12px;font-size:.85rem;font-weight:500}

/* Numbers */
.tc-numbers{display:grid;grid-template-columns:1fr 1fr;gap:8px 20px;margin:12px 0}
.tc-num{display:flex;justify-content:space-between;gap:12px;
  font-variant-numeric:tabular-nums;padding:3px 0;border-bottom:1px dotted var(--line)}
.tc-num-label{color:var(--muted);font-size:.78rem;letter-spacing:.05em}
.tc-num-value{font-weight:600;text-align:right}
.tc-num-sub{color:var(--muted);font-weight:400;font-size:.85em}
.tc-num--stop .tc-num-value{color:var(--red)}
.tc-num--risk-over .tc-num-value{color:var(--amber);font-weight:700}

/* Invalidation */
.tc-invalidation{margin:12px 0 4px;padding:10px 12px;background:var(--bg);
  border-radius:8px;font-size:.9rem}
.tc-invalid,.tc-timestop{padding:2px 0}
.tc-invalid-label{color:var(--muted);letter-spacing:.04em;font-size:.78rem}

/* Gate footer + outcome */
.tc-gate,.tc-outcome{margin-top:10px;padding-top:8px;border-top:1px solid var(--line);
  font-size:.82rem;color:var(--muted)}
.tc-gate-label{font-weight:700;letter-spacing:.05em}
.tc-gate--normal .tc-gate-label{color:var(--green)}
.tc-gate--reduced .tc-gate-label{color:var(--amber)}
.tc-gate--stand-down .tc-gate-label{color:var(--red)}
.tc-outcome{font-weight:600}
.tc-outcome--win{color:var(--green)}
.tc-outcome--loss{color:var(--red)}

/* Tables */
.tc-rej-table,.tc-sc-table{width:100%;border-collapse:collapse;font-size:.85rem}
.tc-rejected th,.tc-scorecard th{text-align:left;color:var(--muted);
  font-weight:600;padding:6px 8px;border-bottom:1px solid var(--line)}
.tc-rejected td,.tc-scorecard td{padding:6px 8px;border-bottom:1px solid var(--line);
  font-variant-numeric:tabular-nums}
.tc-verdict{font-weight:700}
.tc-verdict--pass{color:var(--green)}
.tc-verdict--fail{color:var(--red)}
.tc-verdict--insufficient_n{color:var(--muted)}
.tc-verdict--diagnostic{color:var(--muted);font-weight:500}
.tc-scorecard--greyed{opacity:.6}
.tc-sc-warnings{color:var(--amber);font-size:.82rem;margin:8px 0 0;padding-left:18px}

@media print{
  body{background:#fff}
  .tc-card{page-break-inside:avoid;box-shadow:none;height:48vh}
  .tc-digest{max-width:none}
}
"""
