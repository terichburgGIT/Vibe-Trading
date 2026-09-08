"""M008 -- Journal I/O layer. Phase D, T017.

Persists every Trade Card as an append-only stream of JSON events, one
line per event. Two event kinds share a `card_id`:
  * `"card"`   -- the thesis, written BEFORE the order is submitted (gate
                  step 12 in `Risk_Policy.md` section 5). `outcome` is
                  null at this point. Timestamped with `ts_generated`.
  * `"outcome"` -- appended when the trade closes, carrying realized R,
                   MAE, MFE, capture ratio, slippage, exit reason,
                   adherence. Timestamped with `ts_closed`.

Reads merge the two streams by `card_id`, most recent outcome wins.
Card rows themselves are IMMUTABLE -- there is no `update_card`, and a
second write for the same card_id raises rather than overwriting. This
is `Trade_Card_Spec.md`'s central honesty constraint: "the reasons on it
are predictions, not rationalizations."

Storage choice: JSONL rather than Parquet (the format `03_Modules.md`
M008 nominally names). Reason: Parquet needs pyarrow (~40 MB wheel) for
a benefit -- columnar reads -- that this workload does not have. The
scale of interest is O(hundreds) of trades per year during the paper
phase; JSONL keeps the file trivially inspectable (`less`, `jq`,
`Get-Content`) and removes one dependency from the critical path.
The invariants that actually matter -- append-only, immutable once
written, one row per event, timestamp-ordered -- are properties of the
writer, not the file format, and are enforced here.

Full contract in `03_Modules.md` section M008.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_JOURNAL_PATH = Path.home() / ".vibe-trading" / "journal.jsonl"


class DuplicateCardError(Exception):
    """Raised when write_card is called twice with the same card_id.
    The card row is immutable by design (Trade_Card_Spec.md): a second
    write is either a caller bug or a data-corruption event, never
    "an update."
    """


class MissingCardError(Exception):
    """Raised when patch_outcome references a card_id that was never
    written. An outcome without a card would break the "thesis exists
    before the order" invariant this whole file exists to enforce.
    """


class DuplicateOutcomeError(Exception):
    """Raised when patch_outcome is called twice for the same card_id.
    Outcomes are also immutable once written -- a re-close is a caller
    bug. If a trade genuinely re-enters, that is a new card, not an
    amended outcome.
    """


# --------------------------------------------------------------------------- #
# Write
# --------------------------------------------------------------------------- #


def write_card(
    card: Mapping[str, Any],
    *,
    path: Path | None = None,
    now: datetime | None = None,
) -> str:
    """Write one Trade Card thesis. Returns the `card_id`. Called
    BEFORE the entry order is submitted -- step 12 of Risk_Policy.md
    section 5 rejects any signal whose card was never journaled.

    Required fields: `card_id`, `symbol`, `venue`, `direction`,
    `score`, `sizing`, `levels`. `ts_generated` is set here (UTC ISO
    8601) if the caller didn't set it. `outcome` is forced to null so a
    caller cannot cheat the "thesis first" contract by pre-filling it.
    """
    if not isinstance(card, Mapping):
        raise TypeError("card must be a mapping")
    card_id = card.get("card_id")
    if not card_id or not isinstance(card_id, str):
        raise ValueError("card requires a non-empty string 'card_id'")
    required = ("symbol", "venue", "direction", "score", "sizing", "levels")
    missing = [f for f in required if f not in card]
    if missing:
        raise ValueError(f"card missing required fields: {missing}")

    journal_path = path or DEFAULT_JOURNAL_PATH

    # Refuse a duplicate card_id -- immutability is the point.
    if journal_path.exists():
        for event in _iter_events(journal_path):
            if event.get("kind") == "card" and event.get("card_id") == card_id:
                raise DuplicateCardError(
                    f"card_id {card_id!r} already written; cards are immutable"
                )

    ts = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    row = dict(card)
    row["ts_generated"] = row.get("ts_generated", ts)
    row["outcome"] = None  # invariant: outcome is null at write time
    event = {
        "kind": "card",
        "card_id": card_id,
        "ts_event": ts,
        "card": row,
    }
    _append_event(journal_path, event)
    return card_id


def patch_outcome(
    card_id: str,
    outcome: Mapping[str, Any],
    *,
    path: Path | None = None,
    now: datetime | None = None,
) -> None:
    """Append the closing outcome for `card_id`. Requires the card to
    already exist (MissingCardError otherwise); refuses to patch twice
    (DuplicateOutcomeError otherwise).

    Required outcome fields: `r_multiple` (realized R, signed float),
    `exit_reason` (string). Optional: `mae`, `mfe`, `capture_ratio`,
    `slippage_atr`, `cost_ratio`, `time_in_trade_minutes`, `adherence`
    (bool). No field is silently dropped; the whole outcome mapping is
    appended verbatim.
    """
    if not isinstance(outcome, Mapping):
        raise TypeError("outcome must be a mapping")
    if "r_multiple" not in outcome:
        raise ValueError("outcome requires 'r_multiple'")
    if "exit_reason" not in outcome:
        raise ValueError("outcome requires 'exit_reason'")

    journal_path = path or DEFAULT_JOURNAL_PATH

    if not journal_path.exists():
        raise MissingCardError(
            f"card_id {card_id!r} has no card row -- journal file does not exist"
        )

    seen_card = False
    for event in _iter_events(journal_path):
        if event.get("card_id") != card_id:
            continue
        if event.get("kind") == "card":
            seen_card = True
        elif event.get("kind") == "outcome":
            raise DuplicateOutcomeError(
                f"card_id {card_id!r} already has an outcome; outcomes are immutable"
            )
    if not seen_card:
        raise MissingCardError(
            f"card_id {card_id!r} has no card row -- can't patch a thesis that "
            "was never written (see Trade_Card_Spec.md, 'the card is the journal row')"
        )

    ts = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    event = {
        "kind": "outcome",
        "card_id": card_id,
        "ts_event": ts,
        "outcome": dict(outcome),
    }
    _append_event(journal_path, event)


# --------------------------------------------------------------------------- #
# Read
# --------------------------------------------------------------------------- #


def read_cards(path: Path | None = None) -> list[dict[str, Any]]:
    """Return every card in insertion order, with any patched outcome
    merged in. Cards without an outcome yet come back with
    `outcome=None`. Result is a fresh dict per row -- callers can
    mutate the returned structure without corrupting the journal.
    """
    journal_path = path or DEFAULT_JOURNAL_PATH
    if not journal_path.exists():
        return []

    order: list[str] = []
    cards: dict[str, dict[str, Any]] = {}
    for event in _iter_events(journal_path):
        card_id = event.get("card_id")
        if not card_id:
            continue
        if event.get("kind") == "card":
            row = dict(event["card"])
            cards[card_id] = row
            order.append(card_id)
        elif event.get("kind") == "outcome" and card_id in cards:
            cards[card_id]["outcome"] = dict(event["outcome"])
            cards[card_id]["ts_closed"] = event.get("ts_event")
    return [cards[cid] for cid in order]


def closed_cards(path: Path | None = None) -> list[dict[str, Any]]:
    """Every card whose outcome has been patched -- what the metrics
    layer feeds on. Open cards (outcome=None) are correctly excluded
    from any R-based aggregation, per Metrics_Definitions.md § 6
    ("Counting open positions as wins" is a listed anti-pattern)."""
    return [c for c in read_cards(path) if c.get("outcome") is not None]


# --------------------------------------------------------------------------- #
# Low-level append -- forcibly flushed + fsynced so a crash between the
# write() and the caller returning still leaves a valid journal.
# --------------------------------------------------------------------------- #


def _append_event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, sort_keys=True, default=_json_default) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            # Some filesystems (network mounts, in-memory tmpfs during
            # tests) don't implement fsync; flush is enough there.
            pass


def _iter_events(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"journal {path} line {line_no}: invalid JSON ({exc})"
                ) from exc


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.astimezone(timezone.utc).isoformat()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")
