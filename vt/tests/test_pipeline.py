"""Tests for the M002 -> M005 -> M006 -> M007 composition
(`vt.pipeline.runner`).

These exercise the composition itself, not the modules it composes --
each of those has its own dedicated test file (`test_screen.py`,
`test_rubric.py`, `test_gate.py`, `test_exec_adapter.py`). The
invariants asserted here are only about how the pieces plug together:

  * Universe screen output feeds M005 in the right shape.
  * Ranked eligible scores go through M006's gate.
  * Approved decisions submit atomically via M007 -- in dry_run,
    nothing hits the broker; in live-run, the card is written BEFORE
    the order, and the entry+stop are one atomic pair.
  * Any drift in reconciliation halts the whole pass before any
    signals are even scored.
  * Approvals from earlier in the same run affect the caps the gate
    sees for later signals (open_positions/sector cap increment).
  * `_write_card` is only called when a decision is actually approved
    AND `dry_run=False`.

Run with: pytest vt/tests -m unit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Mapping, Sequence

import pytest

from vt.data.feed import Bar, Quote
from vt.exec.adapter import BrokerExecAdapter, InternalPosition, Position
from vt.gate.calendar import GateState
from vt.indicators.engine import IndicatorFrame
from vt.pipeline import runner
from vt.pipeline.runner import (
    Enrichment,
    PipelineRequest,
    ScoreDecision,
    run_once,
)
from vt.risk.gate import BreakerState
from vt.signal.rubric import Score
from vt.universe.screen import Candidate as UniverseCandidate

pytestmark = pytest.mark.unit


NOW = datetime(2026, 9, 9, 15, 30, 0, tzinfo=timezone.utc)
ASOF = NOW


# --------------------------------------------------------------------------- #
# Fakes -- deliberately minimal, one behavior per knob
# --------------------------------------------------------------------------- #


@dataclass
class FakeFeed:
    """A feed that returns pre-built universe candidates verbatim, plus
    empty bars and a fresh quote per symbol. The pipeline doesn't
    actually inspect the bars we return -- the injected enrichment
    supplies the rubric inputs directly -- so returning `[]` here keeps
    the test focused on composition, not on M002/M003 internals (each
    of which has its own test file).
    """

    universe_by_venue: Mapping[str, Sequence[UniverseCandidate]] = field(default_factory=dict)
    quote_time_by_symbol: Mapping[str, datetime] = field(default_factory=dict)

    def build_universe(
        self,
        symbols: Sequence[str],
        venue: str,
        *,
        asof: datetime,  # noqa: ARG002
    ) -> list[UniverseCandidate]:
        return list(self.universe_by_venue.get(venue, ()))

    def get_bars(
        self,
        symbol: str,  # noqa: ARG002
        timeframe: str = "1d",  # noqa: ARG002
        *,
        start: datetime | None = None,  # noqa: ARG002
        end: datetime | None = None,  # noqa: ARG002
        limit: int = 90,  # noqa: ARG002
    ) -> list[Bar]:
        return []

    def get_quote(self, symbol: str) -> Quote:
        return Quote(
            symbol=symbol,
            bid=100.0,
            ask=100.05,
            last=100.02,
            time=self.quote_time_by_symbol.get(symbol, NOW),
            source_feed="fake",
        )


@dataclass
class FakeBroker:
    """Broker fake that records every submit call and can be told to
    pre-report existing positions (for reconciliation tests). Mirrors
    the fake used in test_exec_adapter but trimmed to what the
    pipeline exercises.
    """

    venue: str
    positions_rows: list[Position] = field(default_factory=list)

    entry_calls: list[dict] = field(default_factory=list)
    stop_calls: list[dict] = field(default_factory=list)
    cancel_calls: list[str] = field(default_factory=list)
    close_calls: list[str] = field(default_factory=list)

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
        self._entry_seq += 1
        oid = f"entry-{self._entry_seq}"
        self.entry_calls.append(
            {
                "symbol": symbol,
                "side": side,
                "size": size,
                "limit_price": limit_price,
                "client_order_id": client_order_id,
                "order_id": oid,
            }
        )
        signed = size if side == "long" else -size
        self.positions_rows.append(Position(venue=self.venue, symbol=symbol, quantity=signed))
        return oid

    def submit_stop(
        self,
        *,
        symbol: str,
        side: Literal["long", "short"],
        size: float,
        stop_price: float,
        client_order_id: str,
    ) -> str:
        self._stop_seq += 1
        oid = f"stop-{self._stop_seq}"
        self.stop_calls.append(
            {
                "symbol": symbol,
                "side": side,
                "size": size,
                "stop_price": stop_price,
                "client_order_id": client_order_id,
                "order_id": oid,
            }
        )
        return oid

    def cancel(self, order_id: str) -> None:
        self.cancel_calls.append(order_id)

    def close_position(self, symbol: str) -> None:
        self.close_calls.append(symbol)
        self.positions_rows = [r for r in self.positions_rows if r.symbol != symbol]

    def positions(self) -> list[Position]:
        return list(self.positions_rows)


# --------------------------------------------------------------------------- #
# Fixture helpers
# --------------------------------------------------------------------------- #


def _universe_candidate(symbol: str, venue: str = "alpaca") -> UniverseCandidate:
    return UniverseCandidate(
        symbol=symbol,
        venue=venue,
        time_of_day_rvol=3.5,
        dollar_volume_today=30_000_000.0,
        price=100.0,
        avg_spread_pct=0.05,
        atr14_pct=2.0,
    )


def _perfect_enrichment(price: float = 100.0, side: str = "long") -> Enrichment:
    """An enrichment that scores 12/12 on the rubric so the pipeline
    always reaches the gate for signals we care about."""
    return Enrichment(
        price=price,
        vwap=price - 1.0,           # R1=2: above VWAP
        ema9=price - 0.5,
        ema21=price - 1.0,          # R1=2: EMA9 > EMA21
        ema9_rising=True,
        ema21_rising=True,
        rsi14=60.0,                 # R2=2: 55 < RSI <= 70
        rvol=3.5,                   # R3=2: RVOL > 3 + rising OBV
        obv_slope=1.0,
        adx14=30.0,                 # R4=2: ADX > 25 + ATR expanding
        atr_expanding=True,
        prior_close=price - 2.0,
        opening_range_high=price - 0.5,
        opening_range_low=price - 1.5,
        broke_opening_range_high=True,  # R5=2
        held_on_retest=True,
        relative_strength_pct=2.0,      # R6=2
        side=side,
        atr_for_stop=1.5,               # entry 100, stop 100 - 1.5*1.5 = 97.75
    )


def _ineligible_enrichment_r1(price: float = 100.0) -> Enrichment:
    """R1=0 (price below VWAP) so the rubric hard-floor rejects even a
    total that would otherwise clear the threshold."""
    e = _perfect_enrichment(price=price)
    return Enrichment(
        price=price,
        vwap=price + 1.0,          # price <= VWAP => R1=0
        ema9=e.ema9, ema21=e.ema21,
        ema9_rising=e.ema9_rising, ema21_rising=e.ema21_rising,
        rsi14=e.rsi14,
        rvol=e.rvol, obv_slope=e.obv_slope,
        adx14=e.adx14, atr_expanding=e.atr_expanding,
        prior_close=e.prior_close,
        opening_range_high=e.opening_range_high,
        opening_range_low=e.opening_range_low,
        broke_opening_range_high=e.broke_opening_range_high,
        held_on_retest=e.held_on_retest,
        relative_strength_pct=e.relative_strength_pct,
        side=e.side, atr_for_stop=e.atr_for_stop,
    )


def _enrich_fn(mapping: Mapping[str, Enrichment]):
    def _fn(uc: UniverseCandidate, bars, indicators, quote) -> Enrichment:  # noqa: ARG001
        return mapping[uc.symbol]
    return _fn


def _normal_calendar() -> GateState:
    return GateState(state="NORMAL", multiplier=1.0, reasons=())


def _standdown_calendar() -> GateState:
    return GateState(state="STAND_DOWN", multiplier=0.0, reasons=("macro_print:FOMC",))


def _request(
    *,
    feed: FakeFeed,
    adapters: Mapping[str, BrokerExecAdapter],
    enrich_map: Mapping[str, Enrichment],
    symbols_by_venue: Mapping[str, Sequence[str]] | None = None,
    calendar_state: GateState | None = None,
    breaker_state: BreakerState | None = None,
    internal_positions: Mapping[str, Sequence[InternalPosition]] | None = None,
    equity: float = 100_000.0,
    dry_run: bool = True,
    journal_path: Path | None = None,
    sector_of=None,
) -> PipelineRequest:
    return PipelineRequest(
        symbols_by_venue=symbols_by_venue or {"alpaca": list(feed.universe_by_venue.get("alpaca", []))},
        asof=ASOF,
        now=NOW,
        equity=equity,
        breaker_state=breaker_state or BreakerState(),
        adapters=adapters,
        calendar_state=calendar_state or _normal_calendar(),
        enrich=_enrich_fn(enrich_map),
        internal_positions=internal_positions or {},
        sector_of=sector_of or (lambda venue, sym: "unknown"),
        journal_path=journal_path,
        feed=feed,
        dry_run=dry_run,
    )


# --------------------------------------------------------------------------- #
# Composition tests
# --------------------------------------------------------------------------- #


def test_run_once_scores_every_universe_candidate() -> None:
    uc1 = _universe_candidate("AAPL")
    uc2 = _universe_candidate("MSFT")
    feed = FakeFeed(universe_by_venue={"alpaca": [uc1, uc2]})
    broker = FakeBroker(venue="alpaca")
    request = _request(
        feed=feed,
        adapters={"alpaca": broker},
        enrich_map={"AAPL": _perfect_enrichment(), "MSFT": _perfect_enrichment()},
        symbols_by_venue={"alpaca": ["AAPL", "MSFT"]},
    )

    outcome = run_once(request)

    assert len(outcome.scores) == 2
    assert {s.symbol for s in outcome.scores} == {"AAPL", "MSFT"}
    assert all(s.total == 12 for s in outcome.scores)


def test_dry_run_never_touches_broker_or_journal(tmp_path: Path) -> None:
    """The default rollout mode. Gate still runs -- the caller can see
    every Decision -- but nothing hits the wire and no card is
    persisted."""
    uc = _universe_candidate("AAPL")
    feed = FakeFeed(universe_by_venue={"alpaca": [uc]})
    broker = FakeBroker(venue="alpaca")
    journal = tmp_path / "journal.jsonl"
    request = _request(
        feed=feed,
        adapters={"alpaca": broker},
        enrich_map={"AAPL": _perfect_enrichment()},
        symbols_by_venue={"alpaca": ["AAPL"]},
        journal_path=journal,
        dry_run=True,
    )

    outcome = run_once(request)

    (row,) = outcome.decisions
    assert row.decision is not None
    assert row.decision.status == "approved"
    assert row.receipt is None
    assert row.card_id is None
    assert broker.entry_calls == []
    assert broker.stop_calls == []
    assert not journal.exists()


def test_live_run_writes_card_before_submitting_atomic_order(tmp_path: Path) -> None:
    """The Risk_Policy.md Sec5 step 12 invariant: card is journaled
    BEFORE the order goes out, so if the submit crashes there is still
    a thesis row on disk to reconcile against."""
    uc = _universe_candidate("AAPL")
    feed = FakeFeed(universe_by_venue={"alpaca": [uc]})
    broker = FakeBroker(venue="alpaca")
    journal = tmp_path / "journal.jsonl"
    request = _request(
        feed=feed,
        adapters={"alpaca": broker},
        enrich_map={"AAPL": _perfect_enrichment()},
        symbols_by_venue={"alpaca": ["AAPL"]},
        journal_path=journal,
        dry_run=False,
    )

    outcome = run_once(request)

    (row,) = outcome.decisions
    assert row.decision is not None
    assert row.decision.status == "approved"
    assert row.card_id is not None
    assert row.receipt is not None
    assert row.receipt.status == "submitted"
    assert row.receipt.entry_order_id is not None
    assert row.receipt.stop_order_id is not None

    # Both legs went out.
    assert len(broker.entry_calls) == 1
    assert len(broker.stop_calls) == 1
    # And the same card_id is the entry's client_order_id (so entry+stop
    # + journal all key off one string, per submit_atomic's contract).
    assert broker.entry_calls[0]["client_order_id"] == row.card_id
    assert broker.stop_calls[0]["client_order_id"] == f"{row.card_id}-stop"

    # Journal was written and contains exactly the one card row.
    assert journal.exists()
    contents = journal.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 1
    import json
    event = json.loads(contents[0])
    assert event["kind"] == "card"
    assert event["card_id"] == row.card_id
    assert event["card"]["symbol"] == "AAPL"
    assert event["card"]["outcome"] is None


def test_drift_halts_before_any_signal_is_scored(tmp_path: Path) -> None:
    """Risk_Policy.md Sec3: adapter disagreement is the most important
    breaker. If broker truth != internal state, we do NOT go on to
    scan for setups -- we stop and page the human."""
    uc = _universe_candidate("AAPL")
    feed = FakeFeed(universe_by_venue={"alpaca": [uc]})
    # Broker reports a position we don't know about -> drift.
    broker = FakeBroker(
        venue="alpaca",
        positions_rows=[Position(venue="alpaca", symbol="MSFT", quantity=42.0)],
    )
    request = _request(
        feed=feed,
        adapters={"alpaca": broker},
        enrich_map={"AAPL": _perfect_enrichment()},
        symbols_by_venue={"alpaca": ["AAPL"]},
        internal_positions={"alpaca": []},
        dry_run=False,
        journal_path=tmp_path / "journal.jsonl",
    )

    outcome = run_once(request)

    assert outcome.halted is True
    assert outcome.halt_reason is not None
    assert "alpaca" in outcome.halt_reason
    assert outcome.scores == ()
    assert outcome.decisions == ()
    # No orders sent -- crucial.
    assert broker.entry_calls == []
    assert broker.stop_calls == []


def test_calendar_stand_down_rejects_every_signal_at_the_gate() -> None:
    """T014 semantics at the composition layer: with multiplier=0.0,
    every ranked eligible score becomes a gate reject with reason
    'calendar_stand_down' -- not silently dropped, still visible on
    the outcome so the caller can render it in the digest."""
    uc = _universe_candidate("AAPL")
    feed = FakeFeed(universe_by_venue={"alpaca": [uc]})
    broker = FakeBroker(venue="alpaca")
    request = _request(
        feed=feed,
        adapters={"alpaca": broker},
        enrich_map={"AAPL": _perfect_enrichment()},
        symbols_by_venue={"alpaca": ["AAPL"]},
        calendar_state=_standdown_calendar(),
        dry_run=False,
    )

    outcome = run_once(request)

    (row,) = outcome.decisions
    assert row.score.entry_eligible is True  # rubric passed
    assert row.decision is not None
    assert row.decision.status == "rejected"
    assert row.decision.reject_reason == "calendar_stand_down"
    assert row.receipt is None
    assert broker.entry_calls == []


def test_rubric_ineligible_signals_never_reach_the_gate() -> None:
    """Ineligible-by-rubric (R1 or R6 hard floor breached, or total
    below threshold) signals are surfaced on the outcome for
    transparency but must NOT go through gate.evaluate -- doing so
    would run size/stop math on a signal we already know we're not
    trading."""
    uc = _universe_candidate("AAPL")
    feed = FakeFeed(universe_by_venue={"alpaca": [uc]})
    broker = FakeBroker(venue="alpaca")
    request = _request(
        feed=feed,
        adapters={"alpaca": broker},
        enrich_map={"AAPL": _ineligible_enrichment_r1()},
        symbols_by_venue={"alpaca": ["AAPL"]},
        dry_run=False,
    )

    outcome = run_once(request)

    (row,) = outcome.decisions
    assert row.score.entry_eligible is False
    assert row.decision is None
    assert row.receipt is None
    assert broker.entry_calls == []


def test_multiple_approvals_increment_open_position_count_for_later_signals() -> None:
    """M006 caps at MAX_CONCURRENT_POSITIONS=3. If we approve four
    signals in one pass, the fourth must see open_positions=3 (from
    the three we already approved earlier this run) and be rejected at
    gate step 5 -- not silently sneak through because the broker
    hadn't reported the fills yet at pipeline start."""
    from vt.risk.gate import MAX_CONCURRENT_POSITIONS

    # Distinct symbols so rank order is deterministic on symbol asc.
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    ucs = [_universe_candidate(s) for s in symbols]
    feed = FakeFeed(universe_by_venue={"alpaca": ucs})
    broker = FakeBroker(venue="alpaca")

    # Each symbol in its own sector so the sector cap (2) doesn't trip
    # before the concurrent cap (3) does.
    sector_by_symbol = {s: f"sector-{s}" for s in symbols}
    request = _request(
        feed=feed,
        adapters={"alpaca": broker},
        enrich_map={s: _perfect_enrichment() for s in symbols},
        symbols_by_venue={"alpaca": symbols},
        dry_run=False,
        sector_of=lambda venue, sym: sector_by_symbol[sym],
    )

    outcome = run_once(request)

    approved = [row for row in outcome.decisions if row.decision and row.decision.status == "approved"]
    rejected = [row for row in outcome.decisions if row.decision and row.decision.status == "rejected"]

    assert len(approved) == MAX_CONCURRENT_POSITIONS
    assert len(rejected) == 1
    assert rejected[0].decision.reject_reason == "max_concurrent_positions"
    # And the broker actually saw the three approved entries.
    assert len(broker.entry_calls) == MAX_CONCURRENT_POSITIONS


def test_sector_cap_increments_within_a_single_run() -> None:
    """Sector cap is 2 (equities). If two approvals in the same sector
    land, the third same-sector signal must be rejected at step 6."""
    symbols = ["AAA", "BBB", "CCC"]
    ucs = [_universe_candidate(s) for s in symbols]
    feed = FakeFeed(universe_by_venue={"alpaca": ucs})
    broker = FakeBroker(venue="alpaca")

    request = _request(
        feed=feed,
        adapters={"alpaca": broker},
        enrich_map={s: _perfect_enrichment() for s in symbols},
        symbols_by_venue={"alpaca": symbols},
        dry_run=False,
        sector_of=lambda venue, sym: "tech",  # everything is one sector
    )

    outcome = run_once(request)

    rejections = [
        row.decision.reject_reason
        for row in outcome.decisions
        if row.decision and row.decision.status == "rejected"
    ]
    assert rejections == ["sector_correlation_cap"]
    approved = [row for row in outcome.decisions if row.decision and row.decision.status == "approved"]
    assert len(approved) == 2


def test_stale_quote_is_rejected_at_gate_step_seven() -> None:
    """Feed hands the pipeline a quote timestamped >5s before `now`;
    gate step 7 (quote_freshness) must reject it. This is the whole
    reason M001's `is_stale` exists -- it needs to be enforced HERE,
    at the composition, not silently trusted from upstream."""
    from vt.risk.gate import MAX_QUOTE_AGE_SECONDS

    uc = _universe_candidate("AAPL")
    stale_time = NOW - timedelta(seconds=MAX_QUOTE_AGE_SECONDS + 1)
    feed = FakeFeed(
        universe_by_venue={"alpaca": [uc]},
        quote_time_by_symbol={"AAPL": stale_time},
    )
    broker = FakeBroker(venue="alpaca")
    request = _request(
        feed=feed,
        adapters={"alpaca": broker},
        enrich_map={"AAPL": _perfect_enrichment()},
        symbols_by_venue={"alpaca": ["AAPL"]},
        dry_run=False,
    )

    outcome = run_once(request)

    (row,) = outcome.decisions
    assert row.decision is not None
    assert row.decision.reject_reason == "quote_freshness"
    assert broker.entry_calls == []


def test_hard_kill_breaker_rejects_before_any_other_gate_step() -> None:
    """A hard-killed breaker state (drawdown or manual kill) must stop
    everything at gate step 1. Composition doesn't care WHY the state
    is killed; the invariant is 'no orders when hard_killed=True'."""
    uc = _universe_candidate("AAPL")
    feed = FakeFeed(universe_by_venue={"alpaca": [uc]})
    broker = FakeBroker(venue="alpaca")
    request = _request(
        feed=feed,
        adapters={"alpaca": broker},
        enrich_map={"AAPL": _perfect_enrichment()},
        symbols_by_venue={"alpaca": ["AAPL"]},
        breaker_state=BreakerState(hard_killed=True),
        dry_run=False,
    )

    outcome = run_once(request)

    (row,) = outcome.decisions
    assert row.decision is not None
    assert row.decision.reject_reason == "kill_switch"
    assert broker.entry_calls == []


def test_ranked_order_is_deterministic_across_runs() -> None:
    """rank() sorts eligible-first, total desc, symbol asc -- the
    composition must preserve that so a daily digest reads the same
    twice. Two eligible signals of equal total, in reverse alpha input
    order, must come back alpha-sorted."""
    ucs = [_universe_candidate("ZZZ"), _universe_candidate("AAA")]
    feed = FakeFeed(universe_by_venue={"alpaca": ucs})
    broker = FakeBroker(venue="alpaca")
    request = _request(
        feed=feed,
        adapters={"alpaca": broker},
        enrich_map={"ZZZ": _perfect_enrichment(), "AAA": _perfect_enrichment()},
        symbols_by_venue={"alpaca": ["ZZZ", "AAA"]},
    )

    outcome = run_once(request)

    assert [s.symbol for s in outcome.scores] == ["AAA", "ZZZ"]


def test_no_universe_no_signals_no_orders_no_halt() -> None:
    """Empty watchlist is a valid pass. It reconciles (no drift), scores
    nothing, decides nothing, submits nothing, and does NOT halt."""
    feed = FakeFeed(universe_by_venue={"alpaca": []})
    broker = FakeBroker(venue="alpaca")
    request = _request(
        feed=feed,
        adapters={"alpaca": broker},
        enrich_map={},
        symbols_by_venue={"alpaca": []},
    )

    outcome = run_once(request)

    assert outcome.halted is False
    assert outcome.scores == ()
    assert outcome.decisions == ()
    assert outcome.universe == {"alpaca": ()}
    assert "alpaca" in outcome.reconciliations
    assert outcome.reconciliations["alpaca"].halted is False


def test_reconcile_result_is_surfaced_on_outcome_even_without_drift() -> None:
    """The outcome carries the full reconciliation per venue so the
    caller can log it / alert on flapping drift even when no halt
    fires. Matching positions -> ReconcileResult(drifts=(), halted=False)."""
    uc = _universe_candidate("AAPL")
    feed = FakeFeed(universe_by_venue={"alpaca": [uc]})
    broker = FakeBroker(
        venue="alpaca",
        positions_rows=[Position(venue="alpaca", symbol="AAPL", quantity=10.0)],
    )
    request = _request(
        feed=feed,
        adapters={"alpaca": broker},
        enrich_map={"AAPL": _perfect_enrichment()},
        symbols_by_venue={"alpaca": ["AAPL"]},
        internal_positions={"alpaca": [InternalPosition(venue="alpaca", symbol="AAPL", quantity=10.0)]},
    )

    outcome = run_once(request)

    assert outcome.halted is False
    result = outcome.reconciliations["alpaca"]
    assert result.halted is False
    assert result.drifts == ()


def test_pre_existing_broker_position_counts_toward_open_positions_cap() -> None:
    """If the broker already reports one open position (reconciled with
    an internal counterpart), then in the same run only 2 further
    approvals can land before MAX_CONCURRENT_POSITIONS trips."""
    from vt.risk.gate import MAX_CONCURRENT_POSITIONS

    symbols = ["AAA", "BBB", "CCC"]  # three new candidates
    ucs = [_universe_candidate(s) for s in symbols]
    feed = FakeFeed(universe_by_venue={"alpaca": ucs})
    # Pre-existing open position on a different symbol.
    broker = FakeBroker(
        venue="alpaca",
        positions_rows=[Position(venue="alpaca", symbol="OLD", quantity=5.0)],
    )
    sector_by_symbol = {"AAA": "s1", "BBB": "s2", "CCC": "s3", "OLD": "s4"}
    request = _request(
        feed=feed,
        adapters={"alpaca": broker},
        enrich_map={s: _perfect_enrichment() for s in symbols},
        symbols_by_venue={"alpaca": symbols},
        internal_positions={"alpaca": [InternalPosition(venue="alpaca", symbol="OLD", quantity=5.0)]},
        dry_run=False,
        sector_of=lambda venue, sym: sector_by_symbol[sym],
    )

    outcome = run_once(request)

    approved = [row for row in outcome.decisions if row.decision and row.decision.status == "approved"]
    rejected = [row for row in outcome.decisions if row.decision and row.decision.status == "rejected"]

    # 3 concurrent max minus 1 already-open = 2 new approvals allowed.
    assert len(approved) == MAX_CONCURRENT_POSITIONS - 1
    assert rejected[-1].decision.reject_reason == "max_concurrent_positions"


def test_multi_venue_universe_and_reconciliation() -> None:
    """Both venues at once. Each venue's reconciliation runs
    independently; universes are separate; scores are ranked across
    both venues; no cross-venue contamination in the fake."""
    aapl = _universe_candidate("AAPL", venue="alpaca")
    btc = _universe_candidate("BTC-USDT", venue="okx")
    feed = FakeFeed(universe_by_venue={"alpaca": [aapl], "okx": [btc]})
    alpaca = FakeBroker(venue="alpaca")
    okx = FakeBroker(venue="okx")
    request = _request(
        feed=feed,
        adapters={"alpaca": alpaca, "okx": okx},
        enrich_map={"AAPL": _perfect_enrichment(), "BTC-USDT": _perfect_enrichment()},
        symbols_by_venue={"alpaca": ["AAPL"], "okx": ["BTC-USDT"]},
        dry_run=False,
    )

    outcome = run_once(request)

    assert set(outcome.universe.keys()) == {"alpaca", "okx"}
    assert set(outcome.reconciliations.keys()) == {"alpaca", "okx"}
    assert {s.symbol for s in outcome.scores} == {"AAPL", "BTC-USDT"}
    approved = [row for row in outcome.decisions if row.decision and row.decision.status == "approved"]
    assert {row.score.symbol for row in approved} == {"AAPL", "BTC-USDT"}
    assert len(alpaca.entry_calls) == 1
    assert len(okx.entry_calls) == 1


def test_decision_receipt_pair_when_atomic_submit_succeeds(tmp_path: Path) -> None:
    """When submit_atomic returns status='submitted', the outcome row
    must carry both the decision and the receipt (with both order ids
    populated). This is the load-bearing contract for a live run --
    caller uses the receipt.entry_order_id/stop_order_id for
    downstream tracking, and card_id ties it back to the journal."""
    uc = _universe_candidate("AAPL")
    feed = FakeFeed(universe_by_venue={"alpaca": [uc]})
    broker = FakeBroker(venue="alpaca")
    request = _request(
        feed=feed,
        adapters={"alpaca": broker},
        enrich_map={"AAPL": _perfect_enrichment()},
        symbols_by_venue={"alpaca": ["AAPL"]},
        journal_path=tmp_path / "journal.jsonl",
        dry_run=False,
    )

    outcome = run_once(request)

    (row,) = outcome.decisions
    assert row.decision.status == "approved"
    assert row.receipt is not None
    assert row.receipt.status == "submitted"
    assert row.receipt.entry_order_id is not None
    assert row.receipt.stop_order_id is not None
    assert row.card_id == row.receipt.client_order_id


def test_atomic_stop_failure_leaves_no_open_position(tmp_path: Path) -> None:
    """T015 at the composition layer. If the stop leg fails after the
    entry filled, submit_atomic runs a compensating flatten and returns
    status='flattened_stop_failed'. The pipeline must surface that
    exact receipt -- callers rely on it to know a card was written
    for a trade that never actually opened."""

    @dataclass
    class StopFailingBroker(FakeBroker):
        def submit_stop(self, **kwargs):  # type: ignore[override]
            raise RuntimeError("broker refused stop leg")

    uc = _universe_candidate("AAPL")
    feed = FakeFeed(universe_by_venue={"alpaca": [uc]})
    broker = StopFailingBroker(venue="alpaca")
    request = _request(
        feed=feed,
        adapters={"alpaca": broker},
        enrich_map={"AAPL": _perfect_enrichment()},
        symbols_by_venue={"alpaca": ["AAPL"]},
        journal_path=tmp_path / "journal.jsonl",
        dry_run=False,
    )

    outcome = run_once(request)

    (row,) = outcome.decisions
    assert row.decision is not None
    assert row.decision.status == "approved"  # gate said yes
    assert row.receipt is not None
    assert row.receipt.status == "flattened_stop_failed"
    # Compensating close ran -- no residual position on the broker.
    assert all(p.symbol != "AAPL" for p in broker.positions())
    # And the card was still written (thesis-first) so the failed
    # attempt has a journal row for post-mortem.
    assert row.card_id is not None
