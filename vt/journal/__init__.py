"""M008 — Journal & Metrics. [Planned], not yet implemented.

Persists every Trade Card, patches outcomes on close, computes
everything in `Metrics_Definitions.md`. Net-new.

Parquet, append-only, never mutate a closed row. Must capture the
intra-trade price path (not just entry/exit) to compute MAE/MFE. Session
stats must be rendered explicitly labelled *diagnostic* — never mistaken
for a verdict (see AD009 and the T027 sample-size guard).

Full contract in `03_Modules.md` section M008.
"""
