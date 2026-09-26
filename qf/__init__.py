"""QuickFlip -- fixed-target crypto scalp with adaptive (k x ATR1h) target,
tighter stop, breakeven arm, and an 8-hour hard time-stop.

A separate strategy from VibeTrading (`vt/`), living beside it in the same
fork only so it can share the venv and the three pieces of infrastructure
worth reusing directly: `vt.data.feed` (fail-loud bars/quotes),
`vt.indicators.engine.atr` (Wilder ATR), and `vt.journal.store`
(append-only, thesis-first journal). It deliberately imports nothing from
VibeTrading's strategy/rubric/risk-gate layers.

Design source of truth: `300_Development/QuickFlip/Strategy_Spec.md` and
`Risk_Policy.md` in the vault (split out of `Scenario_Spec.md`).

Entry points: `python -m qf --help`.
"""
