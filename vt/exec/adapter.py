"""M007 -- Paper Execution Adapter. Phase D, T015 (atomic entry+stop)
and T016 (reconciliation halts on drift).

Submit / cancel / reconcile against Alpaca Paper and OKX/Kraken paper
venues behind one interface. Thin wrapper over upstream connectors
(`agent/src/trading/connectors/*`) -- see AD001, AD012.

Two invariants matter more than anything else in this module, and both
have integration tests (T015, T016) rather than just examples:
  * A filled entry with no resting stop is the worst state the system
    can be in. If stop submission fails after entry succeeded, the
    entry must be flattened before this function returns -- the caller
    must never see a receipt that could correspond to a live unstopped
    position (T015).
  * If broker truth disagrees with internal state on any symbol, halt
    immediately (`Risk_Policy.md` Sec3, "Adapter disagreement"). Do not
    reconcile silently, do not paper over -- the state drift IS the
    signal (T016).

Note: `vt.exec` shadows the `exec` builtin only if imported as a bare
name (`from vt import exec`) -- prefer `from vt.exec import adapter` or
`import vt.exec as vt_exec` everywhere.

Full contract in `03_Modules.md` section M007.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Protocol, Sequence


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OrderRequest:
    """A single atomic entry+stop submission. `client_order_id` is the
    caller's idempotency key -- the broker adapter is expected to refuse a
    duplicate submission with the same id (Alpaca / OKX both support this
    natively). Sized and priced by M006; venue chosen by the caller.
    """

    symbol: str
    side: Literal["long", "short"]
    size: float
    entry_price: float
    stop_price: float
    venue: str
    client_order_id: str


@dataclass(frozen=True)
class OrderReceipt:
    """The result of one `submit_atomic` call. Exactly one of the three
    statuses always holds:
      * `submitted`           -- entry AND stop both live on the broker.
      * `entry_rejected`      -- entry submit failed; nothing on the book.
      * `flattened_stop_failed` -- entry filled, stop submit raised, and
        this module has already issued a compensating close. `error`
        carries the stop failure. If the compensating close ALSO failed,
        `error` carries both messages -- see T015.
    """

    client_order_id: str
    status: Literal["submitted", "entry_rejected", "flattened_stop_failed"]
    entry_order_id: str | None
    stop_order_id: str | None
    error: str | None


@dataclass(frozen=True)
class Position:
    """Broker-reported open position, one row per symbol per venue."""

    venue: str
    symbol: str
    quantity: float  # signed: long > 0, short < 0


@dataclass(frozen=True)
class InternalPosition:
    """What this module BELIEVES it holds. Reconciliation compares this
    against `Position` returned by the broker; any disagreement halts.
    """

    venue: str
    symbol: str
    quantity: float


@dataclass(frozen=True)
class Drift:
    """A single (venue, symbol) where broker != internal. Absent symbol on
    either side is represented with `quantity=0.0` -- either 'broker has
    something we don't think we hold' or 'we think we hold something the
    broker never got'. Both flavours halt.
    """

    venue: str
    symbol: str
    internal_quantity: float
    broker_quantity: float


@dataclass(frozen=True)
class ReconcileResult:
    """`halted` is True iff `drifts` is non-empty. Kept as an explicit
    boolean rather than a derived property so tests read as intent
    ('this reconcile halts trading') rather than as arithmetic.
    """

    drifts: tuple[Drift, ...]
    halted: bool


# --------------------------------------------------------------------------- #
# Broker adapter protocol -- what a concrete venue must supply
# --------------------------------------------------------------------------- #


class BrokerExecAdapter(Protocol):
    """Minimal broker surface M007 needs. Duck-typed -- concrete adapters
    for Alpaca / OKX / Kraken live under this Protocol, and tests supply
    fakes with the same shape. Deliberately does NOT expose 'bracket
    order' primitives even where the venue has them (Alpaca does, OKX
    doesn't) -- keeping entry and stop as two calls means the atomicity
    invariant is enforced by THIS module and can be tested without any
    broker-specific machinery.
    """

    venue: str

    def submit_entry(
        self,
        *,
        symbol: str,
        side: Literal["long", "short"],
        size: float,
        limit_price: float,
        client_order_id: str,
    ) -> str: ...

    def submit_stop(
        self,
        *,
        symbol: str,
        side: Literal["long", "short"],
        size: float,
        stop_price: float,
        client_order_id: str,
    ) -> str: ...

    def cancel(self, order_id: str) -> None: ...

    def close_position(self, symbol: str) -> None: ...

    def positions(self) -> list[Position]: ...


# --------------------------------------------------------------------------- #
# Core operations
# --------------------------------------------------------------------------- #


def submit_atomic(adapter: BrokerExecAdapter, request: OrderRequest) -> OrderReceipt:
    """Submit entry + stop atomically. If stop submission fails after the
    entry succeeded, immediately flatten the entry (cancel-if-open,
    close-if-filled) and return `status='flattened_stop_failed'`. The
    caller must never observe a live entry with no resting stop (T015).

    The stop is submitted with a *derived* client_order_id
    (`<client_order_id>-stop`) so the same idempotency guarantee that
    protects the entry from double submission also protects the stop.
    """
    if request.venue != adapter.venue:
        raise ValueError(
            f"OrderRequest venue={request.venue!r} does not match adapter "
            f"venue={adapter.venue!r} -- routing bug, not a broker error."
        )

    stop_coid = f"{request.client_order_id}-stop"

    try:
        entry_order_id = adapter.submit_entry(
            symbol=request.symbol,
            side=request.side,
            size=request.size,
            limit_price=request.entry_price,
            client_order_id=request.client_order_id,
        )
    except Exception as exc:  # noqa: BLE001 -- surface every broker failure
        return OrderReceipt(
            client_order_id=request.client_order_id,
            status="entry_rejected",
            entry_order_id=None,
            stop_order_id=None,
            error=f"submit_entry: {exc!r}",
        )

    try:
        stop_order_id = adapter.submit_stop(
            symbol=request.symbol,
            side=request.side,
            size=request.size,
            stop_price=request.stop_price,
            client_order_id=stop_coid,
        )
    except Exception as stop_exc:  # noqa: BLE001
        # Compensating action -- cancel the resting entry (if still open)
        # AND close the position (if the entry already filled). Both are
        # best-effort; either raising is captured in `error` so the caller
        # sees the full failure chain rather than a partial success.
        errors = [f"submit_stop: {stop_exc!r}"]
        try:
            adapter.cancel(entry_order_id)
        except Exception as cancel_exc:  # noqa: BLE001
            errors.append(f"cancel({entry_order_id}): {cancel_exc!r}")
        try:
            adapter.close_position(request.symbol)
        except Exception as close_exc:  # noqa: BLE001
            errors.append(f"close_position({request.symbol}): {close_exc!r}")
        return OrderReceipt(
            client_order_id=request.client_order_id,
            status="flattened_stop_failed",
            entry_order_id=entry_order_id,
            stop_order_id=None,
            error=" | ".join(errors),
        )

    return OrderReceipt(
        client_order_id=request.client_order_id,
        status="submitted",
        entry_order_id=entry_order_id,
        stop_order_id=stop_order_id,
        error=None,
    )


def cancel(adapter: BrokerExecAdapter, order_id: str) -> None:
    """Cancel one resting order. Errors propagate -- the caller decides
    whether a cancel failure is fatal (usually it is)."""
    adapter.cancel(order_id)


def positions(adapter: BrokerExecAdapter) -> list[Position]:
    """Broker's view of what's open. Never merged with internal state --
    that's `reconcile`'s job."""
    return list(adapter.positions())


def reconcile(
    adapter: BrokerExecAdapter,
    internal: Sequence[InternalPosition],
    *,
    quantity_tolerance: float = 1e-9,
) -> ReconcileResult:
    """Compare broker truth to internal state. Any (venue, symbol) where
    quantity differs by more than `quantity_tolerance` -- INCLUDING
    'present on one side, absent on the other' -- is a drift, and any
    drift halts (`Risk_Policy.md` Sec3 -- adapter disagreement is the
    least glamorous and most important breaker).

    `internal` may contain rows from multiple venues; only rows whose
    `venue` matches this adapter's are compared. Drifts are returned in a
    stable order (venue, symbol) so tests and logs stay deterministic.
    """
    broker_rows = list(adapter.positions())
    broker_by_key: dict[tuple[str, str], float] = {
        (row.venue, row.symbol): row.quantity for row in broker_rows
    }
    internal_by_key: dict[tuple[str, str], float] = {
        (row.venue, row.symbol): row.quantity
        for row in internal
        if row.venue == adapter.venue
    }

    all_keys = set(broker_by_key) | set(internal_by_key)
    drifts: list[Drift] = []
    for key in sorted(all_keys):
        b_qty = broker_by_key.get(key, 0.0)
        i_qty = internal_by_key.get(key, 0.0)
        if abs(b_qty - i_qty) > quantity_tolerance:
            drifts.append(
                Drift(
                    venue=key[0],
                    symbol=key[1],
                    internal_quantity=i_qty,
                    broker_quantity=b_qty,
                )
            )

    return ReconcileResult(drifts=tuple(drifts), halted=bool(drifts))
