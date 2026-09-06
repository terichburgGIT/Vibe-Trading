"""M006 — Risk & Position Sizer. [Planned], not yet implemented.

*** Highest test priority in the project (T010-T014). Tests before code,
no exceptions — see `16_Next_Steps.md` Phase C. ***

Enforces `Risk_Policy.md` in full: computes size, places the stop, runs
the 12-step pre-trade gate, owns the circuit breakers. Net-new, and must
sit IN FRONT OF the upstream pre-trade path, never replace it (AD001).

Property-based tests, not just examples: no input should ever produce a
size exceeding caps, and no stop modification should ever increase
distance-to-stop. Rejection reasons are logged and histogrammed.

Full contract in `03_Modules.md` section M006.
"""
