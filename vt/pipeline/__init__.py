"""Composition module: M002 -> M005 -> M006 -> M007 in one callable pass.

Every deterministic module (universe screen, rubric scorer, risk gate,
execution adapter, journal, discord alerter) has existed and been unit-
tested in isolation since S019/S022/S023, but nothing called them in
sequence. This module is the sequence.

Entry point: ``vt.pipeline.runner.run_once``. See that module's docstring
for the exact composition order and the dry-run / halt semantics.
"""

from vt.pipeline.enrich import EnrichmentError, make_enricher
from vt.pipeline.gate_feeder import GateFeederError, build_gate_state
from vt.pipeline.runner import (
    Enrichment,
    EnrichmentFn,
    PipelineRequest,
    PipelineOutcome,
    ScoreDecision,
    run_once,
)

__all__ = [
    "Enrichment",
    "EnrichmentError",
    "EnrichmentFn",
    "GateFeederError",
    "PipelineRequest",
    "PipelineOutcome",
    "ScoreDecision",
    "build_gate_state",
    "make_enricher",
    "run_once",
]
