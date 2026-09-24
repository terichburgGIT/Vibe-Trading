"""Manual operator tool: run the composed pipeline once against REAL market
data and a REAL broker adapter, in dry-run mode.

This is deliberately NOT a `vt/` module in the T001+ tested-against-fakes
sense (M001-M012 are; this touches live objects on purpose) -- it's the
same kind of standalone operator entry point `vt/alerts/kill.py` is, run
directly rather than imported and composed.

`dry_run=True` (hardcoded here, not a flag) means: real feed reads, real
broker reads (positions, for reconciliation), real enrichment, real
calendar gate -- but no Trade Card is written and no order is submitted.
This is the "reality check" analogous to B3 for M002: confirm no
exceptions, decisions render sensibly, reconciliation is silent.

Internal positions are seeded from the broker's OWN current read (not left
empty) so reconciliation reflects "does this tool's view of what it holds
agree with the broker," which is silent by construction on a fresh run --
NOT a claim that the account is otherwise flat. A demo account pre-seeded
with test balances will legitimately show non-zero `positions()`; seeding
from broker truth is what keeps that from reading as false drift (see
`vt/exec/okx.py` module docstring, and `16_Next_Steps.md` Phase D).

Usage:
    python -m vt.pipeline.live_dry_run --venue okx
    python -m vt.pipeline.live_dry_run --venue okx --symbols BTC-USDT,ETH-USDT

Equities (`--venue alpaca`) is not wired here yet -- it needs a weekday
session and hasn't been live-dry-run at all; this module raises clearly
rather than pretending to support it.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import vt.data.feed as feed_module
from vt.exec.adapter import BrokerExecAdapter, InternalPosition
from vt.pipeline import PipelineRequest, PipelineOutcome, build_gate_state, make_enricher, run_once
from vt.risk.gate import BreakerState

_DEFAULT_SYMBOLS = {
    "okx": ["BTC-USDT", "ETH-USDT", "SOL-USDT"],
}


def _adapter_for(venue: str) -> BrokerExecAdapter:
    if venue == "okx":
        from vt.exec.okx import OKXExecAdapter

        return OKXExecAdapter()
    raise NotImplementedError(
        f"live_dry_run has no wiring for venue={venue!r} yet -- only 'okx' has been "
        "live-verified (see 13_Session_Log.md). Wiring Alpaca needs a weekday session."
    )


def _equity_for(venue: str, adapter: BrokerExecAdapter) -> float:
    if venue == "okx":
        from src.trading.connectors.okx import sdk as okx_sdk

        snapshot = okx_sdk.get_account_snapshot(okx_sdk.load_config())
        if snapshot.get("status") != "ok":
            raise RuntimeError(f"could not read account equity: {snapshot.get('error')}")
        return float(snapshot["account"]["total_equity"])
    raise NotImplementedError(f"no equity read wired for venue={venue!r}")


def run(venue: str, symbols: list[str]) -> PipelineOutcome:
    adapter = _adapter_for(venue)

    broker_positions = adapter.positions()
    internal_positions = {
        venue: [InternalPosition(venue=p.venue, symbol=p.symbol, quantity=p.quantity) for p in broker_positions]
    }
    print(f"Seeded internal state from broker: {len(broker_positions)} position(s)")
    for p in broker_positions:
        print(f"  {p.symbol}: {p.quantity}")

    equity = _equity_for(venue, adapter)
    print(f"\nEquity: {equity:,.2f}")

    now = datetime.now(timezone.utc)

    print(f"\nBuilding live GateState ({venue})...")
    calendar_state = build_gate_state(feed_module, now, venue=venue)
    print(f"  state={calendar_state.state} multiplier={calendar_state.multiplier}")

    enrich = make_enricher(feed_module)

    request = PipelineRequest(
        symbols_by_venue={venue: symbols},
        asof=now,
        now=now,
        equity=equity,
        breaker_state=BreakerState(equity_peak=equity),
        adapters={venue: adapter},
        calendar_state=calendar_state,
        enrich=enrich,
        internal_positions=internal_positions,
        dry_run=True,
    )

    print("\n" + "=" * 70)
    print(f"RUNNING run_once (venue={venue}, dry_run=True)")
    print("=" * 70)
    outcome = run_once(request)

    print(f"\nhalted: {outcome.halted}  halt_reason: {outcome.halt_reason}")

    print("\n--- Reconciliation ---")
    for v, result in outcome.reconciliations.items():
        print(f"  {v}: halted={result.halted}  drifts={len(result.drifts)}")
        for d in result.drifts:
            print(f"    DRIFT {d.symbol}: internal={d.internal_quantity} broker={d.broker_quantity}")

    print("\n--- Universe candidates ---")
    for v, candidates in outcome.universe.items():
        print(f"  {v}: {len(candidates)} candidate(s)")
        for c in candidates:
            print(f"    {c.symbol}")

    print(f"\n--- Scores ({len(outcome.scores)}) ---")
    for s in outcome.scores:
        print(f"  {s.symbol}: total={s.total} eligible={s.entry_eligible} reason={s.ineligible_reason}")

    print(f"\n--- Decisions ({len(outcome.decisions)}) ---")
    for sd in outcome.decisions:
        dec = sd.decision
        if dec is None:
            print(f"  {sd.score.symbol}: no gate run (rubric-ineligible)")
        else:
            print(
                f"  {sd.score.symbol}: status={dec.status} size={dec.size} "
                f"stop={dec.stop_price} reject={dec.reject_reason}"
            )
        print(f"    receipt={sd.receipt}  card_id={sd.card_id}")

    print("\nDone. dry_run=True -- no cards written, no orders submitted.")
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venue", default="okx", choices=["okx", "alpaca"])
    parser.add_argument("--symbols", default=None, help="comma-separated; defaults to a small liquid watchlist")
    args = parser.parse_args()

    symbols = args.symbols.split(",") if args.symbols else _DEFAULT_SYMBOLS.get(args.venue, [])
    if not symbols:
        print(f"no default watchlist for venue={args.venue!r}; pass --symbols", file=sys.stderr)
        return 1

    try:
        run(args.venue, symbols)
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
