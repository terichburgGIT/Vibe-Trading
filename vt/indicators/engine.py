"""M003 — Indicator Engine. [Planned], not yet implemented.

Computes the six rubric inputs (VWAP, EMA9/21, RSI14, OBV, ADX14, ATR14)
plus RVOL. `pandas-ta` wrapper alongside upstream `technical_indicators`.

VWAP must reset at session open for equities and use a rolling 24h
window for crypto — getting this wrong produces plausible-looking but
meaningless numbers, hence the golden-value fixtures in T005.

Full contract in `03_Modules.md` section M003.
"""
