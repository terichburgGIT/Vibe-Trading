"""Tests for M007 -- vt.exec.adapter (T015, T016; see 06_Tests.md).

Phase D (`16_Next_Steps.md`): T015 (entry+stop submit atomically) and
T016 (reconciliation halts on drift) are the two invariants that stand
between an approved risk-gated Decision and an account-ending state.

T015 is specifically the "filled entry with no resting stop is
unreachable" invariant -- a stop-submit failure after an entry fill
MUST trigger a compensating close before this module hands a receipt
back to the caller.

T016 is specifically "broker reports 26 shares, internal state says 13"
-- any (venue, symbol) disagreement, in either direction, halts.

Run with: pytest vt/tests -m unit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pytest

from vt.exec import adapter as vt_exec

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fake broker adapter -- exercises orchestration without touching a live venue
# --------------------------------------------------------------------------- #


@dataclass
class FakeBroker:
    venue: str = "alpaca"
    positions_rows: list[vt_exec.Position] = field(default_factory=list)

    # Submission behaviour toggles
    fail_entry: bool = False
    fail_stop: bool = False
    fail_cancel_after_stop_fail: bool = False
    fail_close_after_stop_fail: bool = False
    entry_leaves_position: bool = True  # simulate immediate fill by default

    # Call recorder
    entry_calls: list[dict] = field(default_factory=list)
    stop_calls: list[dict] = field(default_factory=list)
    cancel_calls: list[str] = field(default_factory=list)
    close_calls: list[str] = field(default_factory=list)

    # Idempotency ledger -- refuse duplicate client_order_id like a real broker.
    _seen_coids: set[str] = field(default_factory=set)

    _entry_seq: int = 0
    _stop_seq: int = 0

    def submit_entry(
        self,
        *,
        symbol: str,
        side: Literal["long", "short"],
        size: float,
        limit_price: float,
        client_order_id: str,
    ) -> str:
        if client_order_id in self._seen_coids:
            raise RuntimeError(f"duplicate client_order_id: {client_order_id}")
        self._seen_coids.add(client_order_id)
        if self.fail_entry:
            raise RuntimeError("entry submit refused by broker")
        self._entry_seq += 1
        order_id = f"entry-{self._entry_seq}"
        self.entry_calls.append(
            {
                "symbol": symbol,
                "side": side,
                "size": size,
                "limit_price": limit_price,
                "client_order_id": client_order_id,
                "order_id": order_id,
            }
        )
        # Simulate immediate fill -- position appears on the broker.
        if self.entry_leaves_position:
            signed = size if side == "long" else -size
            self.positions_rows.append(
                vt_exec.Position(venue=self.venue, symbol=symbol, quantity=signed)
            )
        return order_id

    def submit_stop(
        self,
        *,
        symbol: str,
        side: Literal["long", "short"],
        size: float,
        stop_price: float,
        client_order_id: str,
    ) -> str:
        if client_order_id in self._seen_coids:
            raise RuntimeError(f"duplicate client_order_id: {client_order_id}")
        self._seen_coids.add(client_order_id)
        if self.fail_stop:
            raise RuntimeError("stop submit refused by broker")
        self._stop_seq += 1
        order_id = f"stop-{self._stop_seq}"
        self.stop_calls.append(
            {
                "symbol": symbol,
                "side": side,
                "size": size,
                "stop_price": stop_price,
                "client_order_id": client_order_id,
                "order_id": order_id,
            }
        )
        return order_id

    def cancel(self, order_id: str) -> None:
        if self.fail_cancel_after_stop_fail:
            raise RuntimeError(f"cancel {order_id} refused")
        self.cancel_calls.append(order_id)

    def close_position(self, symbol: str) -> None:
        if self.fail_close_after_stop_fail:
            raise RuntimeError(f"close {symbol} refused")
        self.close_calls.append(symbol)
        self.positions_rows = [
            row for row in self.positions_rows if row.symbol != symbol
        ]

    def positions(self) -> list[vt_exec.Position]:
        return list(self.positions_rows)


def _request(**overrides) -> vt_exec.OrderRequest:
    defaults = dict(
        symbol="AAPL",
        side="long",
        size=13.0,
        entry_price=175.0,
        stop_price=172.5,
        venue="alpaca",
        client_order_id="coid-001",
    )
    defaults.update(overrides)
    return vt_exec.OrderRequest(**defaults)


# --------------------------------------------------------------------------- #
# T015 -- Entry and stop submit atomically
# --------------------------------------------------------------------------- #


def test_submit_atomic_happy_path_lands_both_entry_and_stop() -> None:
    broker = FakeBroker()
    receipt = vt_exec.submit_atomic(broker, _request())

    assert receipt.status == "submitted"
    assert receipt.entry_order_id == "entry-1"
    assert receipt.stop_order_id == "stop-1"
    assert receipt.error is None
    assert len(broker.entry_calls) == 1
    assert len(broker.stop_calls) == 1
    # Position was left on the book (stop submission succeeded) -- the
    # caller now correctly believes there is a stopped position out there.
    assert len(broker.positions_rows) == 1
    assert broker.close_calls == []
    assert broker.cancel_calls == []


def test_submit_atomic_entry_rejected_never_touches_stop() -> None:
    broker = FakeBroker(fail_entry=True)
    receipt = vt_exec.submit_atomic(broker, _request())

    assert receipt.status == "entry_rejected"
    assert receipt.entry_order_id is None
    assert receipt.stop_order_id is None
    assert "submit_entry" in (receipt.error or "")
    # Nothing landed on the book, no compensating actions attempted.
    assert broker.stop_calls == []
    assert broker.cancel_calls == []
    assert broker.close_calls == []
    assert broker.positions_rows == []


def test_submit_atomic_stop_failure_triggers_compensating_flatten() -> None:
    """The T015 invariant proper -- 'entry fill with no resting stop is
    unreachable'. If the stop submit raises after entry succeeded, the
    caller must not receive a receipt that could correspond to a live
    unstopped position on the broker."""
    broker = FakeBroker(fail_stop=True)
    receipt = vt_exec.submit_atomic(broker, _request())

    assert receipt.status == "flattened_stop_failed"
    assert receipt.entry_order_id == "entry-1"
    assert receipt.stop_order_id is None
    assert "submit_stop" in (receipt.error or "")
    # Compensating action: cancel the resting entry AND close the position.
    assert broker.cancel_calls == ["entry-1"]
    assert broker.close_calls == ["AAPL"]
    # And critically: the broker no longer has an open position for AAPL.
    assert all(row.symbol != "AAPL" for row in broker.positions_rows)


def test_submit_atomic_stop_failure_captures_secondary_close_failure() -> None:
    """If the compensating close ALSO fails, the receipt must surface
    BOTH failures -- the caller needs to know the flatten did not
    actually complete so a human can be paged."""
    broker = FakeBroker(fail_stop=True, fail_close_after_stop_fail=True)
    receipt = vt_exec.submit_atomic(broker, _request())

    assert receipt.status == "flattened_stop_failed"
    assert "submit_stop" in (receipt.error or "")
    assert "close_position" in (receipt.error or "")


def test_submit_atomic_derives_distinct_stop_coid_for_idempotency() -> None:
    """Both entry and stop must have their own idempotency key so the
    broker's duplicate-refusal ledger protects both legs independently."""
    broker = FakeBroker()
    vt_exec.submit_atomic(broker, _request(client_order_id="abc"))

    (entry,) = broker.entry_calls
    (stop,) = broker.stop_calls
    assert entry["client_order_id"] == "abc"
    assert stop["client_order_id"] == "abc-stop"
    assert entry["client_order_id"] != stop["client_order_id"]


def test_submit_atomic_second_call_with_same_coid_is_rejected_by_broker() -> None:
    """The adapter delegates idempotency to the broker (Alpaca and OKX
    both refuse duplicate client_order_ids natively). A retry with the
    same coid must not double-submit -- it comes back as entry_rejected,
    NOT as a fresh 'submitted' receipt."""
    broker = FakeBroker()
    first = vt_exec.submit_atomic(broker, _request(client_order_id="dup"))
    second = vt_exec.submit_atomic(broker, _request(client_order_id="dup"))

    assert first.status == "submitted"
    assert second.status == "entry_rejected"
    # And there is still exactly one entry+stop pair on the broker.
    assert len(broker.entry_calls) == 1
    assert len(broker.stop_calls) == 1


def test_submit_atomic_venue_mismatch_is_a_routing_bug_not_a_broker_error() -> None:
    """A request routed to the wrong adapter is a programming error --
    it must raise loudly, not silently be treated as an entry_rejected
    (which would look like a broker refusal to the caller / journal)."""
    broker = FakeBroker(venue="alpaca")
    with pytest.raises(ValueError, match="does not match"):
        vt_exec.submit_atomic(broker, _request(venue="okx"))
    # And nothing was submitted to the wrong venue as a side effect.
    assert broker.entry_calls == []


# --------------------------------------------------------------------------- #
# T016 -- Reconciliation halts on drift
# --------------------------------------------------------------------------- #


def test_reconcile_matching_positions_does_not_halt() -> None:
    broker = FakeBroker(
        positions_rows=[vt_exec.Position(venue="alpaca", symbol="AAPL", quantity=13.0)]
    )
    internal = [vt_exec.InternalPosition(venue="alpaca", symbol="AAPL", quantity=13.0)]

    result = vt_exec.reconcile(broker, internal)

    assert result.halted is False
    assert result.drifts == ()


def test_reconcile_quantity_mismatch_halts_immediately() -> None:
    """The T016 canonical case: broker reports 26 shares, internal
    state says 13. Immediate halt, drift record surfaces both numbers."""
    broker = FakeBroker(
        positions_rows=[vt_exec.Position(venue="alpaca", symbol="AAPL", quantity=26.0)]
    )
    internal = [vt_exec.InternalPosition(venue="alpaca", symbol="AAPL", quantity=13.0)]

    result = vt_exec.reconcile(broker, internal)

    assert result.halted is True
    assert len(result.drifts) == 1
    (drift,) = result.drifts
    assert drift.venue == "alpaca"
    assert drift.symbol == "AAPL"
    assert drift.internal_quantity == 13.0
    assert drift.broker_quantity == 26.0


def test_reconcile_broker_has_extra_symbol_halts() -> None:
    """'Broker has something we don't think we hold' -- e.g. a manual
    trade someone placed in the account outside the system. Halts."""
    broker = FakeBroker(
        positions_rows=[vt_exec.Position(venue="alpaca", symbol="TSLA", quantity=5.0)]
    )
    internal: list[vt_exec.InternalPosition] = []

    result = vt_exec.reconcile(broker, internal)

    assert result.halted is True
    (drift,) = result.drifts
    assert drift.symbol == "TSLA"
    assert drift.internal_quantity == 0.0
    assert drift.broker_quantity == 5.0


def test_reconcile_internal_has_extra_symbol_halts() -> None:
    """'We think we hold something the broker never got' -- e.g. a stop
    fired and we never processed the fill event. Halts."""
    broker = FakeBroker(positions_rows=[])
    internal = [vt_exec.InternalPosition(venue="alpaca", symbol="MSFT", quantity=10.0)]

    result = vt_exec.reconcile(broker, internal)

    assert result.halted is True
    (drift,) = result.drifts
    assert drift.symbol == "MSFT"
    assert drift.internal_quantity == 10.0
    assert drift.broker_quantity == 0.0


def test_reconcile_ignores_other_venues_internal_state() -> None:
    """An OKX row in `internal` must not be flagged as drift against an
    Alpaca adapter -- reconciliation is per-venue, and the caller is
    expected to run one reconcile per adapter."""
    broker = FakeBroker(venue="alpaca", positions_rows=[])
    internal = [
        vt_exec.InternalPosition(venue="okx", symbol="BTC-USDT", quantity=0.1),
    ]

    result = vt_exec.reconcile(broker, internal)

    assert result.halted is False
    assert result.drifts == ()


def test_reconcile_tolerance_absorbs_fractional_share_rounding() -> None:
    """Fractional-share brokers (Alpaca) return 8-decimal quantities.
    A sub-tolerance rounding gap must NOT halt trading -- but anything
    above tolerance must."""
    broker = FakeBroker(
        positions_rows=[
            vt_exec.Position(venue="alpaca", symbol="AAPL", quantity=13.000000001)
        ]
    )
    internal = [vt_exec.InternalPosition(venue="alpaca", symbol="AAPL", quantity=13.0)]

    within = vt_exec.reconcile(broker, internal, quantity_tolerance=1e-6)
    assert within.halted is False

    strict = vt_exec.reconcile(broker, internal, quantity_tolerance=1e-12)
    assert strict.halted is True


def test_reconcile_multiple_drifts_returned_in_stable_order() -> None:
    """Deterministic order matters for logs, tests, and any downstream
    consumer that hashes the drift list."""
    broker = FakeBroker(
        positions_rows=[
            vt_exec.Position(venue="alpaca", symbol="TSLA", quantity=1.0),
            vt_exec.Position(venue="alpaca", symbol="AAPL", quantity=1.0),
            vt_exec.Position(venue="alpaca", symbol="MSFT", quantity=1.0),
        ]
    )
    internal: list[vt_exec.InternalPosition] = []

    result = vt_exec.reconcile(broker, internal)

    assert [d.symbol for d in result.drifts] == ["AAPL", "MSFT", "TSLA"]


# --------------------------------------------------------------------------- #
# Small surface-area checks -- cancel / positions passthroughs
# --------------------------------------------------------------------------- #


def test_cancel_delegates_to_adapter() -> None:
    broker = FakeBroker()
    vt_exec.cancel(broker, "some-order-id")
    assert broker.cancel_calls == ["some-order-id"]


def test_positions_returns_broker_view_unmodified() -> None:
    rows = [
        vt_exec.Position(venue="alpaca", symbol="AAPL", quantity=13.0),
        vt_exec.Position(venue="alpaca", symbol="MSFT", quantity=-5.0),
    ]
    broker = FakeBroker(positions_rows=list(rows))
    assert vt_exec.positions(broker) == rows
