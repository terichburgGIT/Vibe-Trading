"""M002 — Universe Builder. [Planned], not yet implemented.

Daily screen producing <=20 eligible names per venue per
`Strategy_Spec.md` section 1. Net-new (no upstream equivalent).

Time-of-day-aware RVOL is the core requirement — the baseline is
volume-by-this-time-of-day over the trailing 20 sessions, never a
full-day average (that's the classic bug that makes the screen useless
before noon; T004 exists specifically to catch it).

Full contract in `03_Modules.md` section M002.
"""
