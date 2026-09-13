"""M002 -> M005 -> M006 -> M007 in a single call.

Every deterministic module now exists and is tested in isolation
(S013-S023 in `13_Session_Log.md`); this file is the one place where the
whole chain finally runs top-to-bottom. It is what stands between the
current state and a first paper order.

Pipeline order, per `Strategy_Spec.md` / `Risk_Policy.md`::

    for each venue in request.symbols_by_venue:
        reconcile(adapter, internal_positions[venue])   # M007, halt on drift
    if any venue drifted:
        return (halted, no signals, no orders)          # Risk_Policy Sec3
    for each venue's universe candidates (M002):
        fetch bars + quote via `feed`                   # M001
        compute indicators (M003)
        enrich(...) -> M005.Candidate                   # R5/R6 filler is
                                                        # injected: this
                                                        # module does not
                                                        # invent OR/HL,
                                                        # prior close, or
                                                        # benchmark math
    rank all rubric candidates (M005)
    for each eligible score, in rank order:
        build Signal (calendar_multiplier from M004,
                      quote_age from feed, open/sector
                      positions from broker)
        gate.evaluate (M006)                            # 12-step gate
        if approved and not dry_run:
            journal.write_card                          # M008, satisfies
                                                        # gate step 12's
                                                        # "card_written"
                                                        # contract; done
                                                        # BEFORE the order
            exec.submit_atomic (M007)                   # atomic entry+stop
        increment running open/sector position counts   # so the next
                                                        # signal in this
                                                        # same run sees
                                                        # the updated state

Design choices worth stating explicitly:

  * **Dependency-injected everywhere.** `feed`, `adapters`,
    `calendar_state`, and `enrich` all come in on the request. This lets
    the composition be unit-tested with in-memory fakes (`test_pipeline`)
    without ever touching a real broker or a live market. It is also
    what will let the same `run_once` drive a live paper session and a
    replay-mode E2E dry run (T008) from the same code path.

  * **Enrichment is injected, not derived.** M002's `Candidate` carries
    universe stats (RVOL, dollar volume, price, spread, ATR%). M005's
    `Candidate` needs pre-computed indicator values *plus* R5 structural
    facts (prior close, opening-range high/low, break+retest hold) and
    R6 relative strength vs. a benchmark. The R5/R6 fields are not in
    M002 or M003 output; there is no data source wired up for them yet.
    Rather than half-solving that here with fake defaults, the caller
    supplies an `EnrichmentFn` that maps `(UniverseCandidate, bars,
    IndicatorFrame, quote) -> Enrichment`. The pipeline is complete once
    a real enrichment exists; today it lets the composition be tested
    exhaustively with a deterministic fake enrichment.

  * **Card is written BEFORE submit, but only for signals that would be
    submitted.** Risk_Policy.md Sec5 step 12 rejects any signal whose
    Trade Card was never journaled. That check is satisfied by writing
    the card immediately before `submit_atomic`, not by writing cards
    for signals we do not intend to trade (which would pollute the
    journal with unfired-thesis rows the metrics layer would then have
    to filter). The gate is called with `card_written=True` as a
    tentative claim; if the gate approves and we are not in dry_run, we
    write the card and then submit. Dry-run leaves the journal
    untouched.

  * **Halt on drift is decisive.** If any adapter's reconciliation shows
    drift, the whole pipeline stops before any signals are scored --
    per Risk_Policy.md Sec3, adapter disagreement is the least glamorous
    and most important breaker, and continuing to scan for setups when
    we do not agree with the broker about what we already hold is
    exactly the wrong call. The outcome carries `halted=True` and
    `halt_reason` names which venue drifted.

  * **No new AD001 exceptions.** This module does not edit any upstream
    file. It composes existing public APIs from vt.universe / vt.signal
    / vt.risk / vt.exec / vt.gate / vt.journal / vt.data / vt.indicators.

Full contract in `03_Modules.md`; test coverage in
`vt/tests/test_pipeline.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence
import uuid

from vt.data.feed import Bar, Quote
from vt.data.feed import get_bars as default_get_bars
from vt.data.feed import get_quote as default_get_quote
from vt.exec.adapter import (
    BrokerExecAdapter,
    InternalPosition,
    OrderReceipt,
    OrderRequest,
    ReconcileResult,
    reconcile as adapter_reconcile,
    submit_atomic,
)
from vt.gate.calendar import GateState
from vt.indicators.engine import IndicatorFrame, compute as compute_indicators
from vt.journal import store as journal_store
from vt.risk.gate import (
    BreakerState,
    Decision,
    Signal,
    evaluate as gate_evaluate,
)
from vt.signal.rubric import (
    Candidate as RubricCandidate,
    Score,
    rank as rank_scores,
)
from vt.universe.screen import Candidate as UniverseCandidate, build_universe


# --------------------------------------------------------------------------- #
# Injected dependencies -- protocols instead of concrete imports so tests can
# swap in in-memory fakes without touching a live broker / live market.
# --------------------------------------------------------------------------- #


class FeedProtocol(Protocol):
    """The subset of `vt.data.feed` the pipeline needs. `vt.data.feed`
    itself satisfies this (get_bars / get_quote are module-level
    functions); tests supply a fake with the same shape.
    """

    def get_bars(  # noqa: D401 -- protocol shape mirrors vt.data.feed
        self,
        symbol: str,
        timeframe: str = "1d",
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 90,
    ) -> list[Bar]: ...

    def get_quote(self, symbol: str) -> Quote: ...


class _DefaultFeed:
    """Adapter around `vt.data.feed`'s module-level functions so it
    satisfies `FeedProtocol` (which expects methods on an object).
    """

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1d",
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 90,
    ) -> list[Bar]:
        return default_get_bars(symbol, timeframe, start=start, end=end, limit=limit)

    def get_quote(self, symbol: str) -> Quote:
        return default_get_quote(symbol)


@dataclass(frozen=True)
class Enrichment:
    """The pre-computed rubric inputs a caller-supplied `EnrichmentFn`
    returns for one universe candidate. Kept as a plain dataclass rather
    than a full `RubricCandidate` because `symbol`/`venue` come from the
    universe candidate the pipeline already has -- the enrichment only
    supplies the fields M003 and the R5/R6 filler would compute.

    R5 structural facts (`prior_close`, `opening_range_high/low`,
    `broke_opening_range_high`, `held_on_retest`) and R6
    `relative_strength_pct` are the fields M002+M003 do NOT currently
    produce -- an enrichment function is where a caller wires in prior-
    session data and a benchmark comparator without this composition
    module having to know how those are computed.
    """

    price: float
    vwap: float
    ema9: float
    ema21: float
    ema9_rising: bool
    ema21_rising: bool
    rsi14: float
    rvol: float
    obv_slope: float
    adx14: float
    atr_expanding: bool
    prior_close: float
    opening_range_high: float
    opening_range_low: float
    broke_opening_range_high: bool
    held_on_retest: bool
    relative_strength_pct: float
    # Signal-level fields the enrichment also chooses per candidate --
    # side (long/short) and the ATR used for stop-distance sizing (which
    # is 1.5x ATR14 on the 5-min timeframe per Risk_Policy.md Sec2, but
    # this module doesn't fetch 5-min bars separately; the caller passes
    # the right value in).
    side: str  # "long" | "short"
    atr_for_stop: float


EnrichmentFn = Callable[
    [UniverseCandidate, Sequence[Bar], IndicatorFrame, Quote],
    Enrichment,
]
"""Turn one (universe candidate, bars, indicators, quote) tuple into the
full rubric inputs. Injected so the pipeline is complete today with a
test-supplied deterministic enricher, and remains complete tomorrow with
a real one that wires in prior-session data + benchmark comparison.
"""


SectorFn = Callable[[str, str], str]
"""(venue, symbol) -> sector label. Optional; if omitted, every symbol
is treated as sector 'unknown' and the sector cap (M006 step 6) counts
symbols by that single label. `_default_sector` supplies this fallback.
"""


def _default_sector(venue: str, symbol: str) -> str:  # noqa: ARG001
    return "unknown"


# --------------------------------------------------------------------------- #
# Request / outcome data types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PipelineRequest:
    """One pipeline invocation. Everything the composition needs -- data
    sources, adapters, calendar state, current equity, breaker state --
    lives here so `run_once` is a pure function of its inputs (which is
    what makes it testable without any real broker or real market feed).
    """

    symbols_by_venue: Mapping[str, Sequence[str]]
    """Watchlist per venue, e.g. {"alpaca": ["AAPL", ...], "okx":
    ["BTC-USDT", ...]}. Empty venues are allowed and skipped."""

    asof: datetime
    """Session timestamp for universe screening and calendar gating."""

    now: datetime
    """Wall-clock reference for breaker time-boxed halts and quote-age
    checks. Split from `asof` so tests can advance `now` independently
    of the market timestamp (and so historical replays can use a
    realistic `asof` with a real `now`)."""

    equity: float
    """Account equity in the base currency -- feeds M006's
    fixed-fractional sizer."""

    breaker_state: BreakerState
    """Current circuit-breaker state. Persisted separately by the
    caller (see `vt.risk.gate.save_breaker_state`); the pipeline reads
    it but does not update it -- breaker updates are driven by trade
    outcomes via `record_trade`, which is a post-close event outside
    this pass."""

    adapters: Mapping[str, BrokerExecAdapter]
    """Venue -> concrete adapter (`vt.exec.alpaca.AlpacaExecAdapter`,
    or a fake for tests). Must have one entry per venue in
    `symbols_by_venue`."""

    calendar_state: GateState
    """M004 output for this session. The pipeline does not call
    `gate_state()` itself because the required inputs (VIX, realized
    vol, index returns) are not something this module owns; the caller
    computes them and passes the resolved `GateState` in."""

    enrich: EnrichmentFn
    """See `EnrichmentFn` above."""

    internal_positions: Mapping[str, Sequence[InternalPosition]] = field(
        default_factory=dict
    )
    """Venue -> what THIS system believes it holds. Compared against
    broker truth in `reconcile`. Missing venues default to []
    (equivalent to 'nothing open')."""

    sector_of: SectorFn = _default_sector
    """(venue, symbol) -> sector label for M006's sector cap."""

    max_quote_age_seconds: float = 5.0
    """Passed straight through to the M006 `quote_age_seconds` gate
    input. `Risk_Policy.md`'s 5s ceiling is the default."""

    journal_path: Path | None = None
    """Where to append Trade Cards. `None` uses M008's default at
    `~/.vibe-trading/journal.jsonl`."""

    feed: FeedProtocol | None = None
    """Optional feed override for tests. `None` -> `vt.data.feed`."""

    dry_run: bool = True
    """True (the default, and how first-paper-trade rollout will start)
    means the pipeline runs the whole gate and produces `Decision`s but
    does NOT write cards and does NOT submit orders. False means write
    cards for approved decisions and submit them."""


@dataclass(frozen=True)
class ScoreDecision:
    """One rubric score paired with the gate decision that resulted. If
    the score was ineligible (R1/R6 hard floors, or total below
    threshold), `decision` is None -- the gate is not run on ineligible
    scores. If eligible, `decision` is the M006 verdict; `receipt` is
    the M007 receipt when the decision was approved AND we submitted
    (i.e. not dry_run), else None.
    """

    score: Score
    decision: Decision | None
    receipt: OrderReceipt | None
    card_id: str | None
    """The card_id journal.write_card returned when a card was written
    for this decision, else None."""


@dataclass(frozen=True)
class PipelineOutcome:
    """Full record of one `run_once` pass. Everything that happened is
    on the outcome -- nothing is written to logs or global state that
    isn't also here, so tests can assert against a single value.
    """

    universe: Mapping[str, tuple[UniverseCandidate, ...]]
    reconciliations: Mapping[str, ReconcileResult]
    halted: bool
    halt_reason: str | None
    scores: tuple[Score, ...]
    """Every scored candidate in `rank()` order (eligible-first, then
    total desc, then symbol asc). Ineligible scores appear too --
    knowing which candidates were considered and rejected is as
    important as knowing which were approved."""
    decisions: tuple[ScoreDecision, ...]
    """One entry per score. `.decision`/`.receipt` are None for entries
    the gate was never run on (ineligible-by-rubric) and `.receipt` is
    also None for entries that were rejected/halted by the gate or
    would have been submitted in a live run but weren't due to
    `dry_run=True`."""


# --------------------------------------------------------------------------- #
# The composition itself
# --------------------------------------------------------------------------- #


def run_once(request: PipelineRequest) -> PipelineOutcome:
    """Run one M002 -> M005 -> M006 -> M007 pass; see module docstring."""
    feed = request.feed if request.feed is not None else _DefaultFeed()

    # ------------------------------------------------------------------ #
    # 1) Universe (M002) per venue
    # ------------------------------------------------------------------ #
    # `build_universe` reaches into `vt.data.feed` internally; tests can
    # bypass it via `_screen_via` on the feed (see test module). Here we
    # only use `build_universe` when no test-supplied override exists.
    # For the composition test, feeds also expose a `build_universe`
    # method; if not present we fall back to the real one.
    universe: dict[str, tuple[UniverseCandidate, ...]] = {}
    for venue, symbols in request.symbols_by_venue.items():
        if not symbols:
            universe[venue] = ()
            continue
        build_fn = getattr(feed, "build_universe", None)
        if build_fn is not None:
            candidates = build_fn(list(symbols), venue, asof=request.asof)
        else:
            candidates = build_universe(list(symbols), venue, asof=request.asof)
        universe[venue] = tuple(candidates)

    # ------------------------------------------------------------------ #
    # 2) Reconcile (M007) per venue -- halt whole pipeline on any drift
    # ------------------------------------------------------------------ #
    reconciliations: dict[str, ReconcileResult] = {}
    drifted_venues: list[str] = []
    for venue, adapter in request.adapters.items():
        internal = request.internal_positions.get(venue, ())
        result = adapter_reconcile(adapter, list(internal))
        reconciliations[venue] = result
        if result.halted:
            drifted_venues.append(venue)

    if drifted_venues:
        return PipelineOutcome(
            universe=universe,
            reconciliations=reconciliations,
            halted=True,
            halt_reason=f"broker_drift:{','.join(sorted(drifted_venues))}",
            scores=(),
            decisions=(),
        )

    # ------------------------------------------------------------------ #
    # 3) Enrich each universe candidate into a M005 Candidate
    # ------------------------------------------------------------------ #
    rubric_candidates: list[tuple[RubricCandidate, UniverseCandidate, Enrichment, Quote]] = []
    for venue, candidates in universe.items():
        for uc in candidates:
            bars = feed.get_bars(uc.symbol, "5m", end=request.asof, limit=390)
            indicators = compute_indicators(bars)
            quote = feed.get_quote(uc.symbol)
            enrichment = request.enrich(uc, bars, indicators, quote)
            rc = RubricCandidate(
                symbol=uc.symbol,
                venue=uc.venue,
                price=enrichment.price,
                vwap=enrichment.vwap,
                ema9=enrichment.ema9,
                ema21=enrichment.ema21,
                ema9_rising=enrichment.ema9_rising,
                ema21_rising=enrichment.ema21_rising,
                rsi14=enrichment.rsi14,
                rvol=enrichment.rvol,
                obv_slope=enrichment.obv_slope,
                adx14=enrichment.adx14,
                atr_expanding=enrichment.atr_expanding,
                prior_close=enrichment.prior_close,
                opening_range_high=enrichment.opening_range_high,
                opening_range_low=enrichment.opening_range_low,
                broke_opening_range_high=enrichment.broke_opening_range_high,
                held_on_retest=enrichment.held_on_retest,
                relative_strength_pct=enrichment.relative_strength_pct,
            )
            rubric_candidates.append((rc, uc, enrichment, quote))

    # ------------------------------------------------------------------ #
    # 4) Score + rank (M005)
    # ------------------------------------------------------------------ #
    scores_only = rank_scores([rc for rc, _, _, _ in rubric_candidates])

    # Index enrichment lookups by (venue, symbol) so we can pair a
    # ranked score back to the quote/enrichment/universe row it came
    # from. Symbols are unique per venue per pass.
    lookup: dict[tuple[str, str], tuple[UniverseCandidate, Enrichment, Quote]] = {
        (rc.venue, rc.symbol): (uc, enr, q)
        for rc, uc, enr, q in rubric_candidates
    }

    # ------------------------------------------------------------------ #
    # 5) Gate + submit each eligible score, in rank order
    # ------------------------------------------------------------------ #
    open_positions_total = sum(
        len([r for r in adapter.positions()])
        for adapter in request.adapters.values()
    )
    sector_counts: dict[str, int] = {}
    for adapter in request.adapters.values():
        for pos in adapter.positions():
            key = request.sector_of(pos.venue, pos.symbol)
            sector_counts[key] = sector_counts.get(key, 0) + 1

    outcome_rows: list[ScoreDecision] = []
    for score in scores_only:
        if not score.entry_eligible:
            outcome_rows.append(
                ScoreDecision(score=score, decision=None, receipt=None, card_id=None)
            )
            continue

        try:
            uc, enrichment, quote = lookup[(score.venue, score.symbol)]
        except KeyError:  # pragma: no cover -- rank never invents symbols
            raise RuntimeError(
                f"internal error: ranked score for {score.venue}/{score.symbol} "
                "has no matching enrichment row"
            )

        quote_age = _quote_age_seconds(quote, request.now)
        sector = request.sector_of(uc.venue, uc.symbol)
        signal = Signal(
            symbol=uc.symbol,
            side=enrichment.side,  # type: ignore[arg-type]
            entry_price=enrichment.price,
            atr=enrichment.atr_for_stop,
            calendar_multiplier=request.calendar_state.multiplier,
            rubric_score=score.total,
            rubric_r1=score.breakdown["R1"],
            rubric_r6=score.breakdown["R6"],
            quote_age_seconds=quote_age,
            open_positions=open_positions_total,
            sector_positions=sector_counts.get(sector, 0),
            broker_reconciled=True,  # drift already halted above
            card_written=True,  # tentative: satisfied by the write below
        )

        decision = gate_evaluate(
            signal,
            equity=request.equity,
            breaker_state=request.breaker_state,
            now=request.now,
        )

        card_id: str | None = None
        receipt: OrderReceipt | None = None
        if decision.status == "approved":
            if not request.dry_run:
                card_id = _write_card(
                    request=request,
                    universe_candidate=uc,
                    score=score,
                    enrichment=enrichment,
                    decision=decision,
                )
                adapter = request.adapters[uc.venue]
                order_request = OrderRequest(
                    symbol=uc.symbol,
                    side=enrichment.side,  # type: ignore[arg-type]
                    size=decision.size,
                    entry_price=enrichment.price,
                    stop_price=decision.stop_price if decision.stop_price is not None else 0.0,
                    venue=uc.venue,
                    client_order_id=card_id,
                )
                receipt = submit_atomic(adapter, order_request)
            # Whether we actually submitted or not, the running counts
            # should reflect an approved trade so subsequent signals in
            # this same pass do not exceed the caps just because they
            # are being evaluated in the same run as the first approval.
            open_positions_total += 1
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        outcome_rows.append(
            ScoreDecision(
                score=score, decision=decision, receipt=receipt, card_id=card_id
            )
        )

    return PipelineOutcome(
        universe=universe,
        reconciliations=reconciliations,
        halted=False,
        halt_reason=None,
        scores=tuple(scores_only),
        decisions=tuple(outcome_rows),
    )


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _quote_age_seconds(quote: Quote, now: datetime) -> float:
    if quote.time.tzinfo is None:
        # Feed contract (M001 AD003) says quote timestamps are UTC. A
        # naive one is a data-hygiene bug upstream, not ours to paper
        # over; surface a huge age so the gate's quote-freshness step
        # rejects loudly rather than silently trading on it.
        return float("inf")
    reference = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
    return max(0.0, (reference - quote.time).total_seconds())


def _write_card(
    *,
    request: PipelineRequest,
    universe_candidate: UniverseCandidate,
    score: Score,
    enrichment: Enrichment,
    decision: Decision,
) -> str:
    """Assemble and persist one Trade Card, returning its card_id (also
    used as the M007 client_order_id so entry, stop, journal, and any
    later reconciliation all key off the same string). Runs only for
    approved, non-dry_run decisions -- see module docstring for why we
    do not journal signals we do not intend to trade.
    """
    card_id = f"vt-{universe_candidate.venue}-{universe_candidate.symbol}-{uuid.uuid4().hex[:12]}"
    card = {
        "card_id": card_id,
        "symbol": universe_candidate.symbol,
        "venue": universe_candidate.venue,
        "direction": enrichment.side,
        "score": {
            "total": score.total,
            "breakdown": dict(score.breakdown),
            "reasons": list(score.reasons),
        },
        "sizing": {
            "size": decision.size,
            "equity": request.equity,
            "calendar_multiplier": request.calendar_state.multiplier,
        },
        "levels": {
            "entry_price": enrichment.price,
            "stop_price": decision.stop_price,
            "atr_for_stop": enrichment.atr_for_stop,
        },
    }
    return journal_store.write_card(card, path=request.journal_path, now=request.now)
