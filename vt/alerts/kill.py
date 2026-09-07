"""M012 -- Standalone Kill Switch.

Out-of-band flatten script. Runs when the main app is wedged (T023).

Design invariants (Strategy_Spec / Risk_Policy / 03_Modules.md M012):
  * **No shared state** -- imports nothing from `vt.data`, `vt.risk`,
    `vt.exec`, `vt.gate`, `agent.*`, or any other live-app module.
  * **Own credentials path** -- reads `alpaca.json` / `okx.json` directly
    from `~/.vibe-trading/` (or a caller-supplied override), same
    convention as the rest of the project.
  * **Lazy SDK imports** -- alpaca-py / python-okx are imported *inside*
    the concrete adapter loaders, so `import vt.alerts.kill` succeeds on
    a minimal Python with only the deps for the venue you're targeting.
  * **Cancel orders before closing positions** -- otherwise a resting
    stop can fire mid-flatten and re-open exposure.
  * **Errors are isolated** -- one bad venue does not stop the others;
    one bad order/position does not stop the rest of its venue.

Usage::

    python -m vt.alerts.kill              # flatten every configured venue
    python -m vt.alerts.kill --dry-run    # report what would happen, do nothing
    python -m vt.alerts.kill --creds-dir /path/to/creds
    python -m vt.alerts.kill --no-venues  # smoke-test the entry point
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence


DEFAULT_CREDENTIALS_DIR = Path.home() / ".vibe-trading"


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VenueResult:
    venue: str
    cancelled_orders: tuple[str, ...]
    closed_positions: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class KillReport:
    dry_run: bool
    results: tuple[VenueResult, ...]

    @property
    def total_cancelled(self) -> int:
        return sum(len(r.cancelled_orders) for r in self.results)

    @property
    def total_closed(self) -> int:
        return sum(len(r.closed_positions) for r in self.results)

    @property
    def had_errors(self) -> bool:
        return any(r.errors for r in self.results)

    def to_dict(self) -> dict:
        return {
            "dry_run": self.dry_run,
            "total_cancelled": self.total_cancelled,
            "total_closed": self.total_closed,
            "had_errors": self.had_errors,
            "results": [
                {
                    "venue": r.venue,
                    "cancelled_orders": list(r.cancelled_orders),
                    "closed_positions": list(r.closed_positions),
                    "errors": list(r.errors),
                }
                for r in self.results
            ],
        }


class BrokerKillAdapter(Protocol):
    """Minimal broker surface a kill script needs. Duck-typed."""

    venue: str

    def list_open_orders(self) -> list[str]: ...
    def list_positions(self) -> list[str]: ...
    def cancel_order(self, order_id: str) -> None: ...
    def close_position(self, symbol: str) -> None: ...


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def kill_all(
    adapters: Sequence[BrokerKillAdapter], *, dry_run: bool = False
) -> KillReport:
    """Cancel every open order and close every position on every adapter."""
    results: list[VenueResult] = []
    for adapter in adapters:
        results.append(_flatten_one(adapter, dry_run=dry_run))
    return KillReport(dry_run=dry_run, results=tuple(results))


def _flatten_one(adapter: BrokerKillAdapter, *, dry_run: bool) -> VenueResult:
    errors: list[str] = []

    try:
        orders = list(adapter.list_open_orders())
    except Exception as exc:  # noqa: BLE001 -- surface every venue-level failure
        return VenueResult(
            venue=adapter.venue,
            cancelled_orders=(),
            closed_positions=(),
            errors=(f"list_open_orders: {exc!r}",),
        )

    cancelled: list[str] = []
    for order_id in orders:
        if dry_run:
            cancelled.append(order_id)
            continue
        try:
            adapter.cancel_order(order_id)
            cancelled.append(order_id)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"cancel_order({order_id}): {exc!r}")

    try:
        positions = list(adapter.list_positions())
    except Exception as exc:  # noqa: BLE001
        errors.append(f"list_positions: {exc!r}")
        positions = []

    closed: list[str] = []
    for symbol in positions:
        if dry_run:
            closed.append(symbol)
            continue
        try:
            adapter.close_position(symbol)
            closed.append(symbol)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"close_position({symbol}): {exc!r}")

    return VenueResult(
        venue=adapter.venue,
        cancelled_orders=tuple(cancelled),
        closed_positions=tuple(closed),
        errors=tuple(errors),
    )


# --------------------------------------------------------------------------- #
# Concrete adapters -- lazy SDK imports
# --------------------------------------------------------------------------- #


def load_alpaca_adapter(
    credentials_dir: Path | None = None,
) -> BrokerKillAdapter:
    """Build a live Alpaca adapter from `<credentials_dir>/alpaca.json`."""
    creds_dir = credentials_dir or DEFAULT_CREDENTIALS_DIR
    creds_path = creds_dir / "alpaca.json"
    if not creds_path.exists():
        raise FileNotFoundError(
            f"Alpaca credentials not found at {creds_path}. Kill switch needs "
            "its own credentials file -- shared state with the main app is a "
            "design defect, not a shortcut."
        )
    with creds_path.open("r", encoding="utf-8") as fh:
        creds = json.load(fh)

    # Lazy import -- do NOT import alpaca-py at module load; it must be
    # possible to import vt.alerts.kill without any broker SDK installed.
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import ClosePositionRequest, GetOrdersRequest

    client = TradingClient(
        api_key=creds["key_id"],
        secret_key=creds["secret_key"],
        paper=bool(creds.get("is_paper", True)),
    )
    return _AlpacaKillAdapter(
        client=client,
        query_order_status_open=QueryOrderStatus.OPEN,
        get_orders_request_cls=GetOrdersRequest,
        close_position_request_cls=ClosePositionRequest,
    )


class _AlpacaKillAdapter:
    venue = "alpaca"

    def __init__(
        self,
        *,
        client,
        query_order_status_open,
        get_orders_request_cls,
        close_position_request_cls,
    ) -> None:
        self._client = client
        self._open_status = query_order_status_open
        self._orders_request = get_orders_request_cls
        self._close_request = close_position_request_cls

    def list_open_orders(self) -> list[str]:
        req = self._orders_request(status=self._open_status)
        rows = self._client.get_orders(filter=req) or []
        return [str(getattr(r, "id", r)) for r in rows]

    def list_positions(self) -> list[str]:
        rows = self._client.get_all_positions() or []
        return [str(getattr(r, "symbol", r)) for r in rows]

    def cancel_order(self, order_id: str) -> None:
        self._client.cancel_order_by_id(order_id)

    def close_position(self, symbol: str) -> None:
        self._client.close_position(symbol)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vt.alerts.kill",
        description="Standalone kill switch. Cancels every open order and "
        "closes every position on every configured venue.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be flattened; do not call the broker.",
    )
    p.add_argument(
        "--no-venues",
        action="store_true",
        help="Skip loading default venues (smoke-test / test hook).",
    )
    p.add_argument(
        "--creds-dir",
        type=Path,
        default=DEFAULT_CREDENTIALS_DIR,
        help=f"Credentials directory (default: {DEFAULT_CREDENTIALS_DIR}).",
    )
    return p


def _load_default_adapters(creds_dir: Path) -> list[BrokerKillAdapter]:
    """Load every venue whose credentials file is present. Best-effort."""
    adapters: list[BrokerKillAdapter] = []
    if (creds_dir / "alpaca.json").exists():
        adapters.append(load_alpaca_adapter(credentials_dir=creds_dir))
    # OKX kill adapter deferred -- python-okx's flatten surface is
    # asymmetric (spot balances vs. futures positions), so it warrants
    # its own module once the OKX side is actively traded.
    return adapters


def main(
    argv: Sequence[str] | None = None,
    *,
    _extra_adapters: Sequence[BrokerKillAdapter] | None = None,
) -> int:
    """CLI entry point. Returns 0 on clean flatten, nonzero on any error."""
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    adapters: list[BrokerKillAdapter] = []
    if not args.no_venues:
        try:
            adapters.extend(_load_default_adapters(args.creds_dir))
        except Exception as exc:  # noqa: BLE001
            print(f"kill: failed to load venues: {exc!r}", file=sys.stderr)
            return 2
    if _extra_adapters:
        adapters.extend(_extra_adapters)

    report = kill_all(adapters, dry_run=args.dry_run)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 1 if report.had_errors else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
