"""Tests for M008 -- vt.journal.store (T017; see 06_Tests.md).

Phase D (`16_Next_Steps.md`): T017 -- card written before order exists.
Journal row with full thesis exists prior to submit; `outcome` null;
ordering verifiable from timestamps.

Trade_Card_Spec.md's central honesty invariant lives in this file:
'the reasons on it are predictions, not rationalizations.' Cards are
immutable once written; outcomes are appended, never in-place; a card
without an outcome shows outcome=None, not a missing key or a synthetic
zero.

Run with: pytest vt/tests -m unit
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vt.journal import store

pytestmark = pytest.mark.unit


def _card(**overrides) -> dict:
    card = {
        "card_id": "VT-20260908-AAPL-001",
        "symbol": "AAPL",
        "venue": "alpaca",
        "direction": "long",
        "score": {"total": 10, "R1": 2, "R2": 1, "R3": 2, "R4": 2, "R5": 2, "R6": 1},
        "levels": {"entry": 178.40, "stop": 176.54, "target_1": 180.26},
        "sizing": {"shares": 26, "notional": 4638.40, "risk_dollars": 48.36},
    }
    card.update(overrides)
    return card


def _outcome(**overrides) -> dict:
    o = {
        "r_multiple": 1.2,
        "exit_reason": "target_1",
        "mae": -0.3,
        "mfe": 1.4,
        "adherence": True,
    }
    o.update(overrides)
    return o


# --------------------------------------------------------------------------- #
# T017 -- card written before order, outcome null at write time
# --------------------------------------------------------------------------- #


def test_write_card_creates_row_with_null_outcome(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    card_id = store.write_card(_card(), path=journal)

    rows = store.read_cards(journal)
    assert len(rows) == 1
    (row,) = rows
    assert row["card_id"] == card_id
    assert row["outcome"] is None
    assert "ts_generated" in row


def test_write_card_forces_outcome_to_none_even_if_caller_prefilled(tmp_path: Path) -> None:
    """A caller trying to sneak a pre-baked outcome into the card row
    would break the whole 'thesis exists before result' invariant.
    Write forces outcome to None regardless of what was passed in.
    """
    journal = tmp_path / "journal.jsonl"
    card = _card()
    card["outcome"] = {"r_multiple": 99.0}  # cheating attempt
    store.write_card(card, path=journal)

    (row,) = store.read_cards(journal)
    assert row["outcome"] is None


def test_write_card_stamps_ts_generated_in_utc_when_caller_omits_it(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    fixed = datetime(2026, 9, 8, 14, 30, 0, tzinfo=timezone.utc)
    store.write_card(_card(), path=journal, now=fixed)

    (row,) = store.read_cards(journal)
    assert row["ts_generated"] == fixed.isoformat()


def test_card_timestamp_precedes_outcome_timestamp(tmp_path: Path) -> None:
    """T017 headline: ordering verifiable from timestamps. The card
    row's ts_generated must precede its outcome's ts_event."""
    journal = tmp_path / "journal.jsonl"
    generated = datetime(2026, 9, 8, 14, 0, 0, tzinfo=timezone.utc)
    closed = generated + timedelta(minutes=42)

    store.write_card(_card(), path=journal, now=generated)
    store.patch_outcome("VT-20260908-AAPL-001", _outcome(), path=journal, now=closed)

    (row,) = store.read_cards(journal)
    assert row["ts_generated"] == generated.isoformat()
    assert row["ts_closed"] == closed.isoformat()
    assert row["ts_generated"] < row["ts_closed"]
    assert row["outcome"]["r_multiple"] == pytest.approx(1.2)


def test_write_card_requires_card_id(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    bad = _card()
    bad.pop("card_id")
    with pytest.raises(ValueError, match="card_id"):
        store.write_card(bad, path=journal)


def test_write_card_requires_full_thesis_fields(tmp_path: Path) -> None:
    """The card row IS the journal row (Trade_Card_Spec.md). If required
    thesis fields are missing at write time, the whole system's ability
    to explain a trade after the fact is broken -- so refuse early."""
    journal = tmp_path / "journal.jsonl"
    bad = _card()
    bad.pop("sizing")
    with pytest.raises(ValueError, match="sizing"):
        store.write_card(bad, path=journal)


# --------------------------------------------------------------------------- #
# Immutability -- second write / second patch is rejected
# --------------------------------------------------------------------------- #


def test_second_write_with_same_card_id_raises(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    store.write_card(_card(), path=journal)
    with pytest.raises(store.DuplicateCardError):
        store.write_card(_card(), path=journal)


def test_patch_outcome_twice_raises(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    store.write_card(_card(), path=journal)
    store.patch_outcome("VT-20260908-AAPL-001", _outcome(), path=journal)
    with pytest.raises(store.DuplicateOutcomeError):
        store.patch_outcome("VT-20260908-AAPL-001", _outcome(), path=journal)


def test_patch_outcome_without_card_raises(tmp_path: Path) -> None:
    """The outcome-before-card path is exactly the failure mode
    Trade_Card_Spec.md's 'card is the journal row' invariant guards
    against."""
    journal = tmp_path / "journal.jsonl"
    with pytest.raises(store.MissingCardError):
        store.patch_outcome("VT-not-a-card", _outcome(), path=journal)


def test_patch_outcome_requires_r_multiple_and_exit_reason(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    store.write_card(_card(), path=journal)
    with pytest.raises(ValueError, match="r_multiple"):
        store.patch_outcome(
            "VT-20260908-AAPL-001", {"exit_reason": "target_1"}, path=journal
        )
    with pytest.raises(ValueError, match="exit_reason"):
        store.patch_outcome(
            "VT-20260908-AAPL-001", {"r_multiple": 1.0}, path=journal
        )


# --------------------------------------------------------------------------- #
# Read shape -- ordering, closed_cards filter, mutation safety
# --------------------------------------------------------------------------- #


def test_read_cards_returns_insertion_order(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    for i in range(5):
        store.write_card(_card(card_id=f"VT-{i:03d}"), path=journal)
    rows = store.read_cards(journal)
    assert [r["card_id"] for r in rows] == [f"VT-{i:03d}" for i in range(5)]


def test_closed_cards_excludes_open_positions(tmp_path: Path) -> None:
    """Metrics_Definitions.md § 6 anti-pattern: 'counting open positions
    as wins.' The metrics layer feeds on closed_cards, and this is why."""
    journal = tmp_path / "journal.jsonl"
    store.write_card(_card(card_id="VT-CLOSED"), path=journal)
    store.write_card(_card(card_id="VT-OPEN"), path=journal)
    store.patch_outcome("VT-CLOSED", _outcome(), path=journal)

    closed = store.closed_cards(journal)
    assert [c["card_id"] for c in closed] == ["VT-CLOSED"]
    assert closed[0]["outcome"]["r_multiple"] == pytest.approx(1.2)


def test_read_returns_fresh_dicts_so_caller_mutations_do_not_corrupt_the_journal(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "journal.jsonl"
    store.write_card(_card(), path=journal)

    rows_first = store.read_cards(journal)
    rows_first[0]["symbol"] = "MUTATED"

    rows_second = store.read_cards(journal)
    assert rows_second[0]["symbol"] == "AAPL"


def test_read_cards_on_missing_journal_returns_empty(tmp_path: Path) -> None:
    assert store.read_cards(tmp_path / "no_such_file.jsonl") == []


def test_journal_survives_a_reopen_between_events(tmp_path: Path) -> None:
    """Append + fsync guarantees the card is durably on disk before
    the caller can proceed to submit an order. Simulated here by
    write -> release handle -> reopen for patch."""
    journal = tmp_path / "journal.jsonl"
    store.write_card(_card(), path=journal)
    # A fresh process would read via read_cards, then patch.
    rows_before_patch = store.read_cards(journal)
    assert len(rows_before_patch) == 1
    assert rows_before_patch[0]["outcome"] is None

    store.patch_outcome("VT-20260908-AAPL-001", _outcome(r_multiple=-0.7), path=journal)
    (row,) = store.read_cards(journal)
    assert row["outcome"]["r_multiple"] == pytest.approx(-0.7)


def test_corrupt_journal_line_raises_with_line_number(tmp_path: Path) -> None:
    """A truncated write from a crashed process should be diagnosable,
    not silently skipped -- if a card row went missing, the whole
    thesis-first contract is broken and the caller needs to know."""
    journal = tmp_path / "journal.jsonl"
    store.write_card(_card(), path=journal)
    with journal.open("a", encoding="utf-8") as fh:
        fh.write("{not-json\n")
    with pytest.raises(ValueError, match="line 2"):
        store.read_cards(journal)
