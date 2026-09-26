"""`python -m qf` -- operator commands.

    plan   [SYMBOL ...] [--lane L]    read-only: target/stop/fee-floor math, no orders
    open   SYMBOL --lane L --manual   buy now, place the OCO (demo only)
           [--allow-below-floor]      open even if the fee floor says no (mechanics test)
    run    [--once]                   manage open positions until none remain
    status                            book, live P&L, alerts, closed-trade tally
    kill   [--dry-run]                cancel every qf OCO, sell every qf position, halt
    resume                            clear a halt after reviewing why it happened

Run from `VibeTrading/Source` with the project venv.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Sequence

from qf.broker import BrokerError
from qf.config import LANES, QFConfig
from qf.engine import QFError, QuickFlip
from qf.plan import PlanError, TradePlan
from qf.state import StateError

#: Failures an operator should read as a sentence, not a traceback.
_OPERATIONAL_ERRORS = (QFError, PlanError, StateError, BrokerError)


def _engine(cfg: QFConfig) -> QuickFlip:
    from qf.broker import OKXSpotBroker, load_okx_config

    okx_cfg, source = load_okx_config(cfg.okx_config_path)
    return QuickFlip(cfg, OKXSpotBroker(okx_cfg, credentials_source=source, quote_ccy=cfg.quote_ccy))


def _fmt_plan(plan: TradePlan) -> str:
    verdict = "PASS" if plan.passes_fee_floor else "BELOW FEE FLOOR"
    return (
        f"{plan.symbol:<11} ATR1h {plan.atr_frac:6.2%} | target {plan.target_frac:6.2%}  stop {plan.stop_frac:6.2%}"
        f"  drag {plan.drag_frac:5.2%} | net R:R {plan.net_rr:5.2f}  breakeven WR {plan.breakeven_win_rate:4.0%}"
        f" | {plan.lane} ${plan.notional_usd:,.0f}: risk ${plan.planned_risk_usd:,.2f} | {verdict}"
    )


def cmd_plan(qf: QuickFlip, args: argparse.Namespace) -> int:
    symbols = args.symbols or list(qf.cfg.watchlist)
    status = 0
    for symbol in symbols:
        try:
            print(_fmt_plan(qf.plan(symbol, args.lane)))
        except Exception as exc:  # noqa: BLE001 -- report per symbol, keep going
            print(f"{symbol:<11} ERROR {exc}", file=sys.stderr)
            status = 1
    print(f"\nk={qf.cfg.k_atr}  stop_ratio={qf.cfg.stop_ratio}  min_net_rr={qf.cfg.min_net_rr}  (no orders placed)")
    return status


def cmd_open(qf: QuickFlip, args: argparse.Namespace) -> int:
    pos = qf.open(args.symbol, args.lane, manual=args.manual, allow_below_floor=args.allow_below_floor)
    print(
        f"opened {pos.trade_id}: {pos.size} {pos.symbol} @ {pos.entry_px}  "
        f"target {pos.target_px}  stop {pos.stop_px}  arm at {pos.arm_px}  "
        f"deadline {pos.deadline:%Y-%m-%d %H:%M} UTC  (OCO {pos.algo_id})"
    )
    print("start the monitor so the arm and time-stop run: python -m qf run")
    return 0


def cmd_run(qf: QuickFlip, args: argparse.Namespace) -> int:
    for drift in qf.reconcile():
        print(f"DRIFT {drift}", file=sys.stderr)
    while True:
        for event in qf.tick():
            print(f"{time.strftime('%H:%M:%S')} {event}", flush=True)
        book = qf.book()
        if args.once or not book.positions:
            if not book.positions:
                print("no open positions")
            return 1 if book.halted else 0
        time.sleep(qf.cfg.poll_seconds)


def cmd_status(qf: QuickFlip, args: argparse.Namespace) -> int:
    book = qf.book()
    print(f"credentials: {qf.broker.credentials_source}  demo={qf.broker.is_demo}")
    print(f"halted: {book.halted}{' -- ' + book.halt_reason if book.halted else ''}  "
          f"consecutive losses: {book.consecutive_losses}/{qf.cfg.consecutive_loss_halt}")
    for pos in book.positions:
        print(f"  {pos.trade_id} {pos.symbol} size {pos.size} entry {pos.entry_px} "
              f"target {pos.target_px} stop {pos.stop_px}{' (armed)' if pos.armed else ''} "
              f"deadline {pos.deadline:%m-%d %H:%M} UTC  oco={pos.algo_id}")
    for alert in book.alerts:
        print(f"  ALERT {alert}")
    for label, row in qf.closed_summary().items():
        exp = "n/a" if row["expectancy_r"] is None else f"{row['expectancy_r']:+.2f}R"
        print(f"closed {label}: n={row['n']} wins={row['wins']} net ${row['net_pnl_usd']:+,.2f} expectancy {exp}")
    return 0


def cmd_kill(qf: QuickFlip, args: argparse.Namespace) -> int:
    report = qf.kill(dry_run=args.dry_run)
    prefix = "would " if args.dry_run else ""
    print(f"{prefix}close: {list(report.closed)}\n{prefix}cancel OCOs: {list(report.cancelled_algos)}")
    for err in report.errors:
        print(f"ERROR {err}", file=sys.stderr)
    return 1 if report.errors else 0


def cmd_resume(qf: QuickFlip, args: argparse.Namespace) -> int:
    qf.resume()
    print("halt cleared, alerts and loss streak reset")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qf", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("plan", help="show target/stop/fee math, no orders")
    sp.add_argument("symbols", nargs="*")
    sp.add_argument("--lane", default="small", choices=sorted(LANES))
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("open", help="open a position now (demo only)")
    sp.add_argument("symbol")
    sp.add_argument("--lane", required=True, choices=sorted(LANES))
    sp.add_argument("--manual", action="store_true", help="required until the entry signal exists")
    sp.add_argument("--allow-below-floor", action="store_true")
    sp.set_defaults(func=cmd_open)

    sp = sub.add_parser("run", help="manage open positions")
    sp.add_argument("--once", action="store_true")
    sp.set_defaults(func=cmd_run)

    sub.add_parser("status").set_defaults(func=cmd_status)

    sp = sub.add_parser("kill", help="flatten everything QuickFlip owns and halt")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_kill)

    sub.add_parser("resume").set_defaults(func=cmd_resume)
    return p


def main(argv: Sequence[str] | None = None, *, engine: QuickFlip | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        qf = engine or _engine(QFConfig())
        return args.func(qf, args)
    except _OPERATIONAL_ERRORS as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nstopped -- open positions stay protected by their exchange OCOs", file=sys.stderr)
        return 130
