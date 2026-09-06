"""M005 — Signal Scorer. [Planned], not yet implemented.

Applies the 6-component rubric, produces a 0-12 score plus per-component
breakdown, ranks candidates. Net-new. Equal weights, locked until M010
exists (AD007).

This is the module that actually generates the daily trade hypothesis —
deterministically, from the rubric — not the LLM (vt.analyst, M011,
which only narrates an already-scored candidate; see AD005). The
per-component breakdown this module emits is what the Trade Card
displays as "the reasons for the pick."

Full contract in `03_Modules.md` section M005.
"""
