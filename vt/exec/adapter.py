"""M007 — Paper Execution Adapter. [Planned], not yet implemented.

Submit/cancel/reconcile against Alpaca Paper and OKX/Kraken paper venues
behind one interface. Thin wrapper over upstream connectors
(`src/trading/connectors/*`) — see AD001, AD012.

Entry + stop must submit atomically — a filled entry with no resting
stop is the worst state the system can be in. Reconciliation loop
compares broker truth to internal state and halts on drift
(`Risk_Policy.md` section 3).

Note: `vt.exec` shadows the `exec` builtin only if imported as a bare
name (`from vt import exec`) — prefer `from vt.exec import adapter` or
`import vt.exec as vt_exec` everywhere.

Full contract in `03_Modules.md` section M007.
"""
