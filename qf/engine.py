"""QuickFlip orchestration: open, manage, close, kill.

Invariants this module exists to hold (each has an engine test):

  1. **Thesis first.** The journal card is written before the entry order
     is sent (same contract as `vt.journal.store`, Trade_Card_Spec).
  2. **Never unprotected.** A filled entry gets its OCO immediately; if the
     OCO can't be placed, the position is sold at market before `open`
     returns and the book halts. The same rule governs re-arming: if the
     stop can't be amended AND can't be replaced, flatten. A position with
     no OCO is sold on every tick until it is gone.
  3. **Stops only tighten.** The breakeven arm never lowers a stop.
  4. **Sell only what we bought.** Exits sell `Position.size`, never the
     account balance.
  5. **Demo only.** `open` refuses a non-demo OKX profile outright.
  6. **Breaker.** `consecutive_loss_halt` losses in a row halts new opens
     until a human runs `resume`.
  7. **Restart-safe exits.** A market sell's order id is persisted before
     its fill is awaited, and a trade already journaled as closed is never
     closed again -- a crash at any point cannot cause a second sell.
  8. **One writer.** Every read-modify-write of the book holds a lock file,
     so `run` in one terminal and `open` in another cannot drop a position.

Every external dependency (broker, bars, quotes, clock, sleep) is injected
so the whole lifecycle runs under test against fakes.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, ContextManager, Sequence

from qf import manage, state
from qf.broker import AlgoStatus, Broker, Fill
from qf.config import QFConfig
from qf.manage import Action
from qf.plan import FeeModel, TradePlan, atr_frac, build_plan, round_down_to_step
from qf.state import Book, Position
from vt.data import feed
from vt.journal import store as journal

#: Fill polling for market orders (they complete in well under a second).
_FILL_POLL_ATTEMPTS = 10
_FILL_POLL_SECONDS = 0.5
#: Holding may read this much below the recorded size before it counts as
#: drift (exchange-side balance rounding).
_HOLDING_TOLERANCE = 1e-9
#: An exit filling less than this fraction short of the recorded size is
#: treated as complete; anything bigger is surfaced as an alert.
_PARTIAL_FILL_TOLERANCE = 0.001
#: How long `open` waits for a monitor tick to release the book.
_LOCK_TIMEOUT_SECONDS = 60.0
#: clOrdId leg letter per exit reason (entry is "e", OCOs "p"/"a").
_EXIT_LEG = {"time": "t", "protect_failed": "x", "kill": "k"}


class QFError(RuntimeError):
    """A refused or failed QuickFlip action, with an operator-readable reason."""


class FeeFloorError(QFError):
    def __init__(self, plan: TradePlan) -> None:
        super().__init__(
            f"{plan.symbol}: net reward:risk {plan.net_rr:.2f} is below the fee floor "
            f"{plan.min_net_rr:.2f} (target {plan.target_frac:.2%} vs {plan.drag_frac:.2%} round-trip drag; "
            f"needs a {plan.breakeven_win_rate:.0%} win rate to break even). "
            "Pass --allow-below-floor to open anyway as a mechanics test."
        )
        self.plan = plan


@dataclass(frozen=True)
class KillReport:
    closed: tuple[str, ...]
    cancelled_algos: tuple[str, ...]
    errors: tuple[str, ...]


def make_trade_id(prefix: str, symbol: str, now: datetime) -> str:
    """OKX clOrdId charset is alphanumeric, <= 32. prefix + base + UTC
    second stamp leaves room for a leg letter and an attempt number."""
    base = re.sub(r"[^A-Za-z0-9]", "", symbol.split("-")[0])[:10]
    return f"{prefix}{base}{now.astimezone(timezone.utc):%y%m%d%H%M%S}"


class QuickFlip:
    def __init__(
        self,
        cfg: QFConfig,
        broker: Broker,
        *,
        get_bars: Callable[..., Sequence[feed.Bar]] = feed.get_bars,
        get_quote: Callable[[str], feed.Quote] = feed.get_quote,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = cfg
        self.broker = broker
        self._get_bars = get_bars
        self._get_quote = get_quote
        self._clock = clock
        self._sleep = sleep

    # ------------------------------------------------------------------ #
    # Book I/O
    # ------------------------------------------------------------------ #

    def book(self) -> Book:
        return state.load(self.cfg.state_path)

    def _save(self, book: Book) -> Book:
        state.save(self.cfg.state_path, book)
        return book

    def _locked(self) -> ContextManager[None]:
        return state.locked(self.cfg.state_path, timeout_seconds=_LOCK_TIMEOUT_SECONDS)

    # ------------------------------------------------------------------ #
    # Planning (read-only)
    # ------------------------------------------------------------------ #

    def fee_model(self, symbol: str) -> FeeModel:
        try:
            taker = self.broker.taker_rate(symbol)
        except Exception:  # noqa: BLE001 -- a failed fee read must not block planning
            taker = self.cfg.fallback_taker_rate
        return FeeModel(taker_rate=taker, slippage_frac_per_side=self.cfg.slippage_frac_per_side)

    def plan(self, symbol: str, lane: str) -> TradePlan:
        notional = self.cfg.lane_usd(lane)
        bars = self._get_bars(symbol, self.cfg.atr_timeframe, limit=self.cfg.atr_lookback_bars)
        atr = atr_frac(bars, period=self.cfg.atr_period, now=self._clock())
        return build_plan(symbol, lane, notional, atr=atr, fees=self.fee_model(symbol), cfg=self.cfg)

    # ------------------------------------------------------------------ #
    # Open
    # ------------------------------------------------------------------ #

    def open(self, symbol: str, lane: str, *, manual: bool, allow_below_floor: bool = False) -> Position:
        if not manual:
            raise QFError(
                "automated entries need the signal filter, which is not defined yet "
                "(Strategy_Spec §5) -- use a manual test open"
            )
        if not self.broker.is_demo:
            raise QFError("refusing to trade: OKX profile is not demo (Risk_Policy §4, demo first)")
        with self._locked():
            return self._open_locked(symbol, lane, manual=manual, allow_below_floor=allow_below_floor)

    def _open_locked(self, symbol: str, lane: str, *, manual: bool, allow_below_floor: bool) -> Position:
        book = self.book()
        self._check_capacity(book, symbol, self.cfg.lane_usd(lane))
        plan = self.plan(symbol, lane)
        if not plan.passes_fee_floor and not allow_below_floor:
            raise FeeFloorError(plan)

        inst = self.broker.instrument(symbol)
        quote = self._get_quote(symbol)
        now = self._clock()
        if feed.is_stale(quote, now=now):
            raise QFError(f"{symbol}: quote is stale ({quote.time.isoformat()}), not opening")

        trade_id = make_trade_id(self.cfg.id_prefix, symbol, now)
        ref_px = quote.ask or quote.last or 0.0
        self._write_card(trade_id, plan, ref_px, inst.tick_sz, manual=manual, below_floor=not plan.passes_fee_floor)

        try:
            order_id = self.broker.market_buy_usd(symbol, plan.notional_usd, f"{trade_id}e")
            fill = self._wait_fill(symbol, order_id)
            if not fill.done:
                fill = self._stop_filling(symbol, order_id)
        except Exception as exc:
            self._journal_non_trade(trade_id, "entry_failed", str(exc))
            raise QFError(f"{symbol}: entry failed: {exc}") from exc
        if fill.filled_sz <= 0:
            detail = f"order {order_id} filled nothing ({fill.state})"
            self._journal_non_trade(trade_id, "entry_failed", detail)
            if not fill.done:  # still live after a cancel attempt: it may yet fill
                self._save(book.halt(f"{trade_id}: entry {detail} and is still live -- check the account"))
            raise QFError(f"{symbol}: entry {detail}")

        # From here until the OCO is live the position is unprotected: ANY
        # failure in this window flattens it (invariant 2).
        size = self._sellable_size(symbol, fill, inst.lot_sz)
        pos = Position(
            trade_id=trade_id, symbol=symbol, lane=lane, manual=manual,
            notional_usd=fill.filled_sz * fill.avg_px, size=size, entry_px=fill.avg_px,
            target_px=0.0, stop_px=0.0, breakeven_px=0.0, arm_px=0.0,
            planned_risk_usd=fill.filled_sz * fill.avg_px * plan.net_risk_frac,
            opened_at=now, deadline=now + timedelta(hours=plan.max_hold_hours),
            algo_id=None,
        )
        try:
            if not fill.done:
                raise QFError(f"entry order {order_id} still {fill.state} after a cancel; more may fill unmanaged")
            if size < inst.min_sz:
                raise QFError(f"sellable size {size} is below minSz {inst.min_sz}")
            lv = plan.levels(fill.avg_px, tick_sz=inst.tick_sz)
            pos = replace(
                pos, target_px=lv.target_px, stop_px=lv.stop_px, breakeven_px=lv.breakeven_px, arm_px=lv.arm_px
            )
            algo_id = self.broker.place_oco(symbol, size, lv.target_px, lv.stop_px, f"{trade_id}p")
        except Exception as exc:
            book, event = self._exit_now(book, pos, reason="protect_failed")
            msg = f"{trade_id}: could not protect the fill ({exc}); {event}"
            self._save(book.halt(msg))
            raise QFError(msg) from exc

        pos = replace(pos, algo_id=algo_id)
        self._save(book.with_position(pos))
        return pos

    def _stop_filling(self, symbol: str, order_id: str) -> Fill:
        """An entry still working after the poll budget: cancel the rest so
        no coins arrive after the OCO is sized, then take the final fill."""
        try:
            self.broker.cancel_order(symbol, order_id)
        except Exception:  # noqa: BLE001 -- it may have completed meanwhile; the re-read decides
            pass
        return self.broker.order_fill(symbol, order_id)

    def _sellable_size(self, symbol: str, fill: Fill, lot_sz: float) -> float:
        """Filled base minus any base-currency fee, lot-rounded down. The
        account balance caps it when readable (guards a fee-accounting
        surprise); a failed balance read must not block protecting the
        fill, so it falls back to the fill-derived size alone."""
        base = symbol.split("-")[0]
        size = fill.filled_sz - (fill.fee if fill.fee_ccy.upper() == base.upper() else 0.0)
        try:
            size = min(size, self.broker.holding(base))
        except Exception:  # noqa: BLE001 -- the cap is advisory; protection is not
            pass
        return round_down_to_step(max(size, 0.0), lot_sz)

    def _check_capacity(self, book: Book, symbol: str, notional: float) -> None:
        if book.halted:
            raise QFError(f"halted: {book.halt_reason} -- run `python -m qf resume` after reviewing")
        if any(p.symbol == symbol for p in book.positions):
            raise QFError(f"{symbol}: already holding a QuickFlip position (one per symbol)")
        if len(book.positions) >= self.cfg.max_concurrent_positions:
            raise QFError(f"at the {self.cfg.max_concurrent_positions}-position concurrency cap")
        open_notional = sum(p.notional_usd for p in book.positions)
        if open_notional + notional > self.cfg.max_open_notional_usd:
            raise QFError(
                f"would put ${open_notional + notional:,.0f} open, over the "
                f"${self.cfg.max_open_notional_usd:,.0f} bankroll cap"
            )

    # ------------------------------------------------------------------ #
    # Manage
    # ------------------------------------------------------------------ #

    def tick(self) -> list[str]:
        """One management pass over every open position. Returns human
        event lines. One position's failure never stops the others; it is
        recorded as a book alert (surfaced by `status`), never dropped."""
        with self._locked():
            book = self.book()
            closed = self._journaled_closed_ids()
            events: list[str] = []
            for pos in book.positions:
                if pos.trade_id in closed:
                    # journaled, then crashed before the book was saved
                    book = book.without(pos.trade_id)
                    event = f"{pos.trade_id}: already journaled as closed; dropped from book"
                else:
                    try:
                        book, event = self._tick_one(book, pos)
                    except Exception as exc:  # noqa: BLE001 -- isolate per position, surface loudly
                        event = f"{pos.trade_id}: tick error {exc!r}"
                        book = book.alert(event)
                if event:
                    events.append(event)
                self._save(book)
            return events

    def _tick_one(self, book: Book, pos: Position) -> tuple[Book, str]:
        now = self._clock()
        if pos.algo_id is None:
            # no OCO: an exit is in flight, or an earlier flatten failed
            reason = pos.exit_reason or ("time" if now >= pos.deadline else "protect_failed")
            return self._exit_now(book, pos, reason=reason)

        st = self.broker.algo_status(pos.algo_id)
        if st.triggered:
            return self._close_from_algo(book, pos, st)
        if st.failed or st.cancelled:
            # the OCO no longer protects the position -- a triggered order
            # that failed, or a cancel we didn't make. Flatten and halt.
            book, event = self._exit_now(book, pos, reason="protect_failed")
            msg = f"{pos.trade_id}: OCO {pos.algo_id} is {st.state}; {event}"
            return book.halt(msg), msg

        quote = self._get_quote(pos.symbol)
        last = quote.last or quote.bid
        if last is None or feed.is_stale(quote, now=now):
            if now >= pos.deadline:  # the time-stop needs no price
                return self._time_exit(book, pos)
            return book, ""

        pos = manage.observe(pos, last)
        action = manage.decide(pos, last, now)
        if action is Action.TIME_EXIT:
            return self._time_exit(book, pos)
        if action is Action.ARM:
            return self._arm(book, pos)
        return book.with_position(pos), ""

    def _arm(self, book: Book, pos: Position) -> tuple[Book, str]:
        new_stop = manage.armed_stop(pos)
        try:
            self.broker.amend_stop(pos.symbol, pos.algo_id, new_stop)
            armed = replace(pos, armed=True, stop_px=new_stop)
            return book.with_position(armed), f"{pos.trade_id}: armed, stop -> {new_stop}"
        except Exception as amend_exc:  # noqa: BLE001 -- fall back to cancel+replace
            amend_error = amend_exc

        st = self.broker.algo_status(pos.algo_id)
        if st.triggered:
            return self._close_from_algo(book, pos, st)
        self.broker.cancel_algo(pos.symbol, pos.algo_id)  # raises -> retried next tick, OCO still live
        pos = replace(pos, algo_id=None)
        book = self._save(book.with_position(pos))  # checkpoint: unprotected until replaced
        try:
            algo_id = self.broker.place_oco(pos.symbol, pos.size, pos.target_px, new_stop, f"{pos.trade_id}a")
        except Exception as exc:  # noqa: BLE001
            book, event = self._exit_now(book, pos, reason="protect_failed")
            msg = f"{pos.trade_id}: re-arm failed (amend: {amend_error}; replace: {exc}); {event}"
            return book.halt(msg), msg
        armed = replace(pos, armed=True, stop_px=new_stop, algo_id=algo_id)
        return book.with_position(armed), f"{pos.trade_id}: armed via replace, stop -> {new_stop}"

    def _time_exit(self, book: Book, pos: Position) -> tuple[Book, str]:
        try:
            self.broker.cancel_algo(pos.symbol, pos.algo_id)
        except Exception:  # noqa: BLE001 -- maybe it just fired; check before anything else
            st = self.broker.algo_status(pos.algo_id)
            if st.triggered:
                return self._close_from_algo(book, pos, st)
            raise
        return self._exit_now(book, replace(pos, algo_id=None), reason="time")

    # ------------------------------------------------------------------ #
    # Close
    # ------------------------------------------------------------------ #

    def _exit_now(self, book: Book, pos: Position, *, reason: str) -> tuple[Book, str]:
        """Sell an unprotected position; on failure keep it on the book
        (every tick retries), halt, and say so loudly."""
        book, error = self._exit_unprotected(book, pos, reason=reason)
        if error is None:
            return book, f"{pos.trade_id}: {reason} exit, sold {pos.size}"
        msg = f"{pos.trade_id}: {reason} EXIT NOT DONE ({error}) -- position is UNPROTECTED, retrying every tick"
        return self._save(book.halt(msg)), msg

    def _exit_unprotected(self, book: Book, pos: Position, *, reason: str) -> tuple[Book, str | None]:
        """Market-sell a position that has no live OCO, checkpointed so no
        crash can cause a second sell (invariant 7). Returns (book, None)
        once the close is journaled, else (book still holding pos, why)."""
        pos = replace(pos, algo_id=None, exit_reason=pos.exit_reason or reason)
        book = self._save(book.with_position(pos))
        if pos.exit_order_id is None:
            try:
                self._cancel_orphan_algos(pos)
            except Exception as exc:  # noqa: BLE001 -- an unprotected position outranks a rare orphan
                book = self._save(book.alert(f"{pos.trade_id}: orphan-OCO sweep failed ({exc!r}); selling anyway"))
        try:
            if pos.exit_order_id is None:
                leg = _EXIT_LEG.get(pos.exit_reason, "x")
                order_id = self.broker.market_sell(pos.symbol, pos.size, f"{pos.trade_id}{leg}{pos.exit_attempts}")
                pos = replace(pos, exit_order_id=order_id, exit_attempts=pos.exit_attempts + 1)
                book = self._save(book.with_position(pos))
            fill = self._wait_fill(pos.symbol, pos.exit_order_id)
        except Exception as exc:  # noqa: BLE001 -- reported to the caller, which halts
            return book, repr(exc)
        if fill.filled_sz > 0 and fill.done:
            return self._record_close(book, pos, pos.exit_reason, fill), None
        if fill.done:
            # died unfilled: free the checkpoint so the next attempt re-sells
            book = self._save(book.with_position(replace(pos, exit_order_id=None)))
            return book, f"sell {pos.exit_order_id} filled nothing ({fill.state})"
        return book, f"sell {pos.exit_order_id} still {fill.state}; re-checking next tick"

    def _cancel_orphan_algos(self, pos: Position) -> None:
        """A position with no OCO on the book may still have one live on
        the exchange (crash between placing a replacement and saving it).
        Cancel it before selling, or it could later fire against coins
        QuickFlip no longer holds."""
        for symbol, algo_id in self.broker.tagged_live_algos(pos.trade_id):
            self.broker.cancel_algo(symbol, algo_id)

    def _close_from_algo(self, book: Book, pos: Position, st: AlgoStatus) -> tuple[Book, str]:
        if st.ord_id is None:
            raise QFError(f"{pos.trade_id}: OCO {pos.algo_id} triggered but reports no order id yet")
        fill = self._wait_fill(pos.symbol, st.ord_id)
        # actualSide should always be set once triggered; if not, the fill
        # price tells the two legs apart (they sit either side of entry)
        side = st.side or ("tp" if fill.avg_px >= pos.entry_px else "sl")
        if side == "tp":
            reason = "target"
        else:
            reason = "stop_breakeven" if pos.armed else "stop"
        return self._record_close(book, pos, reason, fill), f"{pos.trade_id}: {reason} @ {fill.avg_px}"

    def _record_close(self, book: Book, pos: Position, reason: str, fill: Fill) -> Book:
        fee_quote = fill.fee if fill.fee_ccy.upper() == self.cfg.quote_ccy else fill.fee * fill.avg_px
        row = manage.outcome(
            pos, exit_reason=reason, exit_px=fill.avg_px, exit_size=fill.filled_sz,
            exit_fee_quote=fee_quote, closed_at=self._clock(),
        )
        try:
            journal.patch_outcome(pos.trade_id, row, path=self.cfg.journal_path, now=self._clock())
        except journal.DuplicateOutcomeError:
            return self._save(book.without(pos.trade_id).alert(
                f"{pos.trade_id}: outcome was already journaled; dropped from book without re-recording"
            ))
        losses = book.consecutive_losses + 1 if row["pnl_usd"] < 0 else 0
        book = replace(book.without(pos.trade_id), consecutive_losses=losses)
        if fill.filled_sz < pos.size * (1.0 - _PARTIAL_FILL_TOLERANCE):
            book = book.alert(
                f"{pos.trade_id}: only {fill.filled_sz} of {pos.size} {pos.symbol} sold on {reason}; "
                "remainder is still in the account, unmanaged"
            )
        if losses >= self.cfg.consecutive_loss_halt and not book.halted:
            book = book.halt(f"{losses} consecutive losses (last: {pos.trade_id}, {reason})")
        return self._save(book)  # right after the journal write: shrink the crash window

    def _wait_fill(self, symbol: str, order_id: str) -> Fill:
        fill = self.broker.order_fill(symbol, order_id)
        for _ in range(_FILL_POLL_ATTEMPTS):
            if fill.done:
                break
            self._sleep(_FILL_POLL_SECONDS)
            fill = self.broker.order_fill(symbol, order_id)
        return fill

    # ------------------------------------------------------------------ #
    # Journal
    # ------------------------------------------------------------------ #

    def _write_card(self, trade_id: str, plan: TradePlan, ref_px: float, tick_sz: float, *, manual: bool, below_floor: bool) -> None:
        ref = plan.levels(ref_px, tick_sz=tick_sz) if ref_px > 0 else None
        journal.write_card(
            {
                "card_id": trade_id,
                "strategy": "quickflip",
                "symbol": plan.symbol,
                "venue": self.broker.venue,
                "direction": "long",
                "score": "manual" if manual else "signal",
                "manual": manual,
                "below_fee_floor_override": below_floor,
                "sizing": {"lane": plan.lane, "notional_usd": plan.notional_usd},
                "levels": None if ref is None else {
                    "reference_px": ref.entry_px, "target": ref.target_px, "stop": ref.stop_px,
                    "arm": ref.arm_px, "breakeven": ref.breakeven_px,
                },
                "plan": {
                    "atr_frac": plan.atr_frac, "target_frac": plan.target_frac, "stop_frac": plan.stop_frac,
                    "drag_frac": plan.drag_frac, "net_rr": plan.net_rr,
                    "breakeven_win_rate": plan.breakeven_win_rate, "taker_rate": plan.taker_rate,
                    "k_atr": self.cfg.k_atr, "max_hold_hours": plan.max_hold_hours,
                },
            },
            path=self.cfg.journal_path,
            now=self._clock(),
        )

    def _journal_non_trade(self, trade_id: str, reason: str, detail: str) -> None:
        journal.patch_outcome(
            trade_id,
            {"r_multiple": 0.0, "exit_reason": reason, "pnl_usd": 0.0, "detail": detail},
            path=self.cfg.journal_path,
            now=self._clock(),
        )

    def _journaled_closed_ids(self) -> set[str]:
        return {c["card_id"] for c in journal.closed_cards(self.cfg.journal_path)}

    # ------------------------------------------------------------------ #
    # Reconcile / kill / resume
    # ------------------------------------------------------------------ #

    def reconcile(self) -> list[str]:
        """Two drift checks, either of which halts:
          * an open position's coins are missing (extra coins are fine --
            VibeTrading, demo seed balances);
          * a journaled entry has neither an outcome nor a book row -- the
            signature of a crash inside `open`, which may have left a
            filled, unprotected position nobody is tracking."""
        with self._locked():
            book = self.book()
            drifts = []
            for pos in book.positions:
                if pos.exit_order_id is not None:
                    continue  # mid-exit: the coins may legitimately be gone
                held = self.broker.holding(pos.symbol.split("-")[0])
                if held + _HOLDING_TOLERANCE < pos.size:
                    drifts.append(f"{pos.trade_id}: holds {held} {pos.symbol}, book says {pos.size}")
            on_book = {p.trade_id for p in book.positions}
            for card in journal.read_cards(self.cfg.journal_path):
                if card.get("outcome") is None and card["card_id"] not in on_book:
                    drifts.append(
                        f"{card['card_id']}: journaled entry has no outcome and no book row -- "
                        f"check {card['symbol']} in the account for an untracked fill"
                    )
            if drifts:
                self._save(book.halt("reconcile drift: " + "; ".join(drifts)))
            return drifts

    def kill(self, *, dry_run: bool = False) -> KillReport:
        """Cancel every QuickFlip OCO and sell every QuickFlip position --
        and only QuickFlip's, by recorded size and by `qf` tag. A position
        whose OCO can't be cancelled is NOT sold (the live OCO could later
        fire against coins it no longer holds) unless the OCO already fired,
        in which case its fill is recorded instead."""
        with self._locked():
            book = self.book()
            closed: list[str] = []
            cancelled: list[str] = []
            errors: list[str] = []
            for pos in book.positions:
                if dry_run:
                    closed.append(pos.trade_id)
                    continue
                if pos.algo_id is not None:
                    try:
                        self.broker.cancel_algo(pos.symbol, pos.algo_id)
                        cancelled.append(pos.algo_id)
                    except Exception as exc:  # noqa: BLE001
                        book, handled = self._kill_race(book, pos, exc, errors)
                        if handled:
                            closed.append(pos.trade_id)
                        continue
                book, error = self._exit_unprotected(book, pos, reason="kill")
                if error is None:
                    closed.append(pos.trade_id)
                else:
                    errors.append(f"sell {pos.trade_id}: {error}")
            try:
                for symbol, algo_id in self.broker.tagged_live_algos(self.cfg.id_prefix):
                    if algo_id in cancelled:
                        continue
                    if not dry_run:
                        self.broker.cancel_algo(symbol, algo_id)
                    cancelled.append(algo_id)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"tagged-algo sweep: {exc!r}")
            if not dry_run:
                self._save(book.halt("kill switch"))
            return KillReport(closed=tuple(closed), cancelled_algos=tuple(cancelled), errors=tuple(errors))

    def _kill_race(self, book: Book, pos: Position, cancel_exc: Exception, errors: list[str]) -> tuple[Book, bool]:
        """Cancel failed during kill: record the OCO's own fill if it fired."""
        try:
            st = self.broker.algo_status(pos.algo_id)
            if st.triggered:
                book, _ = self._close_from_algo(book, pos, st)
                return book, True
        except Exception as exc:  # noqa: BLE001
            errors.append(f"status {pos.algo_id}: {exc!r}")
        errors.append(f"cancel {pos.algo_id}: {cancel_exc!r} -- OCO may still be live, position NOT sold")
        return book, False

    def resume(self) -> Book:
        with self._locked():
            return self._save(replace(self.book(), halted=False, halt_reason="", alerts=(), consecutive_losses=0))

    def closed_summary(self) -> dict[str, Any]:
        """Closed-trade tally, split manual vs signal -- the stopping rule
        (Risk_Policy §5) counts signal trades only."""
        rows = [c for c in journal.closed_cards(self.cfg.journal_path) if c["outcome"].get("exit_reason") != "entry_failed"]
        out: dict[str, Any] = {}
        for label, subset in (("manual", [r for r in rows if r.get("manual")]), ("signal", [r for r in rows if not r.get("manual")])):
            pnls = [r["outcome"].get("pnl_usd", 0.0) for r in subset]
            rs = [r["outcome"].get("r_multiple", 0.0) for r in subset]
            out[label] = {
                "n": len(subset),
                "wins": sum(1 for p in pnls if p > 0),
                "net_pnl_usd": sum(pnls),
                "expectancy_r": sum(rs) / len(rs) if rs else None,
            }
        return out
