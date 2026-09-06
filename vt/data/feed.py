"""M001 — Data Layer. [Planned], not yet implemented.

Unified OHLCV + quote access across Alpaca (equities) and OKX/Kraken
(crypto), with yfinance as fallback/history. Normalizes every source to
one bar schema with explicit UTC timestamps and a `source_feed` tag
(AD003 — IEX vs. SIP vs. delayed must be distinguishable downstream).

Thin wrapper over upstream loaders (`src/trading/connectors/*`) — see
AD001. Full contract (signatures, dependencies, test IDs) is in
`03_Modules.md` § M001; do not duplicate that spec here, keep this file
as the implementation once Phase B starts.
"""
