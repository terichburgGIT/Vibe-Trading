"""QuickFlip configuration -- every tunable number in one frozen place.

Defaults marked PROVISIONAL answer an open item from `Scenario_Spec.md` §7
with a data-backed starting value; each is a single field here so changing
it later is a one-line edit, not a refactor. Rationale for every default
lives in `QuickFlip/Strategy_Spec.md` §3 and `Risk_Policy.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

#: State, journal and (optional) dedicated OKX credentials all live here,
#: walled off from VibeTrading's own `~/.vibe-trading/journal.jsonl`.
DEFAULT_STATE_DIR = Path.home() / ".vibe-trading" / "quickflip"

#: Fixed-dollar lanes, picked per trade (Scenario_Spec §2, decided).
LANES: Mapping[str, float] = MappingProxyType(
    {"small": 1_000.0, "medium": 2_000.0, "large": 4_000.0, "xl": 8_000.0}
)


@dataclass(frozen=True)
class QFConfig:
    # --- Target / stop model (Strategy_Spec §3) ---------------------------
    #: PROVISIONAL. target_frac = k_atr x ATR1h%. 2.0 because a 30-day OKX
    #: sample (2026-09-25) put the 75th-percentile 8h up-move at ~2x ATR1h
    #: for BTC, ETH and SOL alike -- a target reachable inside the time box
    #: roughly a quarter of the time on an *unfiltered* entry.
    k_atr: float = 2.0
    #: stop_frac = stop_ratio x target_frac (decided: ~2:1 gross reward:risk).
    stop_ratio: float = 0.5
    #: PROVISIONAL. Move the stop to net breakeven once price has covered
    #: this fraction of the distance to target.
    arm_progress: float = 0.5
    #: Never arm unless price is at least this far above net breakeven --
    #: otherwise the amended stop would sit at/above the last price and
    #: fire immediately.
    arm_min_cushion_frac: float = 0.002
    #: Hard flatten after this many hours (decided).
    max_hold_hours: float = 8.0

    # --- Volatility read ---------------------------------------------------
    atr_period: int = 14
    #: Canonical token understood by the OKX connector's _BAR_MAP. An
    #: unrecognised token silently falls back to daily bars upstream, so
    #: this is validated in `plan.atr_frac` via the bar spacing.
    atr_timeframe: str = "1h"
    atr_lookback_bars: int = 60

    # --- Cost model --------------------------------------------------------
    #: Used only when the live fee-rate read fails. 0.35% is the real OKX
    #: demo taker fee (VibeTrading VT055, re-read live 2026-09-25).
    fallback_taker_rate: float = 0.0035
    #: Per-side slippage allowance on top of the taker fee.
    slippage_frac_per_side: float = 0.001
    #: PROVISIONAL fee floor. A trade must offer at least this net
    #: reward:risk AFTER round-trip drag. 1.0 == breakeven win rate <= 50%,
    #: which is the spec's own "don't need to win most trades" premise
    #: stated honestly (the gross 2:1 collapses once fees are counted).
    min_net_rr: float = 1.0

    # --- Exposure / breaker (Risk_Policy §2) --------------------------------
    max_concurrent_positions: int = 3
    max_open_notional_usd: float = 16_000.0
    consecutive_loss_halt: int = 3

    # --- Plumbing ----------------------------------------------------------
    quote_ccy: str = "USDT"
    #: clOrdId / algoClOrdId prefix -- every QuickFlip order is tagged.
    id_prefix: str = "qf"
    watchlist: tuple[str, ...] = ("BTC-USDT", "ETH-USDT", "SOL-USDT")
    lanes: Mapping[str, float] = field(default_factory=lambda: LANES)
    state_dir: Path = DEFAULT_STATE_DIR
    #: Seconds between monitor ticks in `python -m qf run`.
    poll_seconds: float = 15.0

    @property
    def state_path(self) -> Path:
        return self.state_dir / "state.json"

    @property
    def journal_path(self) -> Path:
        return self.state_dir / "journal.jsonl"

    @property
    def okx_config_path(self) -> Path:
        """Optional dedicated (sub-account) OKX credentials. Absent -> the
        shared `~/.vibe-trading/okx.json`, with isolation by tagging only."""
        return self.state_dir / "okx.json"

    def lane_usd(self, lane: str) -> float:
        try:
            return float(self.lanes[lane])
        except KeyError:
            raise ValueError(f"unknown lane {lane!r}; expected one of {sorted(self.lanes)}") from None
