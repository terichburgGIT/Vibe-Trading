"""M012 — Alerting & Kill Switch. [Planned], not yet implemented.

Discord notifications for fills, breaker trips, drawdown, state drift.
Out-of-band kill script. Uses upstream IM adapters.

`kill_all()` must work when the main process is wedged: standalone entry
point, its own credentials path, no shared state with the rest of the
app (T023 tests this with SIGSTOP on the main process).

Full contract in `03_Modules.md` section M012.
"""
