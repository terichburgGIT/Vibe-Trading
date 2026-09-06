"""VT-001 — our code, kept out of upstream's namespace (AD001).

Rule: nothing in this package edits a file outside `vt/`. Everything here
either wraps an upstream module (`import src....`) or is net-new. If an
upstream edit ever becomes genuinely unavoidable, it must be logged as an
exception under AD001 in `07_Architecture_Decisions.md` before it's made —
see that file's "AD001 exceptions" subsection for the format.

Module map (status, entry point, and what each wraps vs. builds net-new)
lives in `03_Modules.md` at the project root. This package's subpackage
names mirror that map 1:1:

    vt.data        M001  data/feed.py        wraps upstream loaders
    vt.universe    M002  universe/screen.py   net-new
    vt.indicators  M003  indicators/engine.py wraps pandas-ta
    vt.gate        M004  gate/calendar.py     net-new
    vt.signal      M005  signal/rubric.py     net-new
    vt.risk        M006  risk/gate.py         net-new (highest test priority)
    vt.exec        M007  exec/adapter.py      wraps upstream connectors
    vt.journal     M008  journal/             net-new
    vt.render      M009  render/card.py       net-new
    vt.validate    M010  validate/            wraps upstream backtest engines
    vt.analyst     M011  analyst/             advisory only, never execution (AD005)
    vt.alerts      M012  alerts/              wraps upstream IM adapters

Every subpackage is currently empty scaffolding (Phase A5). Implementation
starts at Phase B (see `16_Next_Steps.md`), critical path M001 → M009.
"""
