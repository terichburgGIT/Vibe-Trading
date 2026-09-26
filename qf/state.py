"""Persisted book: open positions plus breaker state, in one JSON file.

The monitor may be restarted at any point during an 8-hour hold (laptop
sleep, crash, deliberate stop). Everything it needs to resume -- which
coins it owns, how many, which exchange order protects them, when the
time-stop falls due -- lives here, written atomically (temp file +
`os.replace`) after every state change. Target and stop are resident on
the exchange as an OCO, so a dead monitor never means an unprotected
position; this file is what lets a restarted one pick the thread back up.

A corrupt file raises rather than reading as "no positions": treating an
unreadable book as empty would let the next `open` stack a second
position on top of an unmanaged one.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

#: A lock file older than this is from a dead process and may be broken.
_STALE_LOCK_SECONDS = 600.0
_LOCK_POLL_SECONDS = 0.2
#: Windows refuses os.replace while any process has the target open (e.g.
#: a concurrent `qf status` read); such a window lasts milliseconds.
_REPLACE_ATTEMPTS = 20
_REPLACE_RETRY_SECONDS = 0.05


class StateError(RuntimeError):
    """The state file exists but cannot be trusted, or is locked by
    another QuickFlip process for longer than we are willing to wait."""


@dataclass(frozen=True)
class Position:
    trade_id: str
    symbol: str
    lane: str
    manual: bool
    #: Quote currency actually spent (filled base x average fill price).
    notional_usd: float
    #: Sellable base quantity: filled, minus any base-currency fee, rounded
    #: down to the instrument's lot size. QuickFlip only ever sells THIS
    #: amount -- never "whatever the account holds" (other systems, and the
    #: demo account's seed balances, may hold the same coin).
    size: float
    entry_px: float
    target_px: float
    stop_px: float
    breakeven_px: float
    arm_px: float
    planned_risk_usd: float
    opened_at: datetime
    deadline: datetime
    #: Exchange OCO protecting the position. None only transiently, while a
    #: time-exit is selling (the OCO was cancelled first).
    algo_id: str | None
    armed: bool = False
    high_px: float = 0.0
    low_px: float = 0.0
    #: Exit checkpoint. Set the moment a market sell is accepted, BEFORE
    #: waiting on its fill, so a restart reads that fill instead of selling
    #: a second time (on a shared account a second sell would dispose of
    #: coins QuickFlip doesn't own).
    exit_order_id: str | None = None
    exit_reason: str = ""
    #: Part of each exit clOrdId, so a retried sell never reuses an id.
    exit_attempts: int = 0

    def to_json(self) -> dict[str, Any]:
        row = asdict(self)
        row["opened_at"] = self.opened_at.isoformat()
        row["deadline"] = self.deadline.isoformat()
        return row

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "Position":
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in row.items() if k in known}
        data["opened_at"] = datetime.fromisoformat(row["opened_at"])
        data["deadline"] = datetime.fromisoformat(row["deadline"])
        return cls(**data)


@dataclass(frozen=True)
class Book:
    positions: tuple[Position, ...] = ()
    consecutive_losses: int = 0
    halted: bool = False
    halt_reason: str = ""
    #: Human-readable problems a tick could not resolve on its own (e.g. a
    #: failed time-exit sell). Surfaced by `status`, cleared by `resume`.
    alerts: tuple[str, ...] = field(default_factory=tuple)

    def with_position(self, pos: Position) -> "Book":
        """Insert or replace (by trade_id)."""
        others = tuple(p for p in self.positions if p.trade_id != pos.trade_id)
        return replace(self, positions=others + (pos,))

    def without(self, trade_id: str) -> "Book":
        return replace(self, positions=tuple(p for p in self.positions if p.trade_id != trade_id))

    def halt(self, reason: str) -> "Book":
        return replace(self.alert(reason), halted=True, halt_reason=reason)

    def alert(self, message: str) -> "Book":
        """Idempotent: a condition re-detected every tick is one alert."""
        if message in self.alerts:
            return self
        return replace(self, alerts=self.alerts + (message,))


def load(path: Path) -> Book:
    if not path.exists():
        return Book()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return Book(
            positions=tuple(Position.from_json(p) for p in raw.get("positions", [])),
            consecutive_losses=int(raw.get("consecutive_losses", 0)),
            halted=bool(raw.get("halted", False)),
            halt_reason=str(raw.get("halt_reason", "")),
            alerts=tuple(raw.get("alerts", [])),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise StateError(f"QuickFlip state at {path} is unreadable ({exc}); fix or move it aside") from exc


def save(path: Path, book: Book) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "positions": [p.to_json() for p in book.positions],
        "consecutive_losses": book.consecutive_losses,
        "halted": book.halted,
        "halt_reason": book.halt_reason,
        "alerts": list(book.alerts),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


@contextmanager
def locked(path: Path, *, timeout_seconds: float = 30.0) -> Iterator[None]:
    """Exclusive lock for a read-modify-write of the book.

    `python -m qf run` loops in one terminal while `open` runs in another;
    without this, a tick that loaded the book before `open` saved would
    write it back without the new position -- leaving a live OCO nobody
    manages. `O_CREAT | O_EXCL` is atomic on every platform we run on.
    """
    lock = path.with_suffix(path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > _STALE_LOCK_SECONDS:
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise StateError(f"QuickFlip state is locked by another process ({lock})") from None
            time.sleep(_LOCK_POLL_SECONDS)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)
