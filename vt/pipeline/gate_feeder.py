"""Live `GateState` feeder for the pipeline (M004 real inputs).

`vt.gate.calendar.gate_state(...)` is a pure function of four numeric
inputs -- `vix`, `realized_vol`, `realized_vol_history`, `index_returns`
-- plus the hand-maintained events YAML. S024's pipeline took a
`GateState` as a literal on the request because no feeder existed to
produce those numbers. This module is that feeder: it derives the
regime-vol inputs from index daily bars via M001 and calls `gate_state`,
so the composition can be driven off live data instead of hand-supplied
literals (Pipeline_Reference.md Sec5 item 2).

What it computes, all from one pull of the index's daily bars:

  * **`index_returns`** -- close-to-close daily returns (decimal fractions,
    matching M004's unit: 0.001 == +0.1%). `gate_state` inspects the last
    element against a 2-sigma move of the preceding window (the REDUCED
    "day after a > 2-sigma index move" trigger).
  * **`realized_vol`** -- the standard deviation of the most recent
    `realized_vol_window` daily returns: a single fraction on the same
    scale as M004's `realized_vol` (~0.015 for a normal index day).
  * **`realized_vol_history`** -- that same rolling stdev computed for each
    prior day, up to `realized_vol_history_len` (60) values, so
    `gate_state` can take its 20th-percentile "dead tape" threshold
    (Strategy_Spec.md Sec2 STAND_DOWN-on-low-realized-vol).

What it does NOT do: fetch VIX. No free VIX source via M001 is settled
(Alpaca's IEX tier does not carry the CBOE VIX index reliably; the VIX
symbol question is unresolved -- see 03_Modules.md M004). `vix` is an
optional injected input; left `None`, `gate_state` simply does not use it
and never spuriously downgrades the state on its absence. Wiring a real
VIX source is a documented follow-up, not this module's job -- keeping it
an explicit injection point is more honest than a fetch that might fail
silently.

Fail-loud policy mirrors `vt.pipeline.enrich`: an empty index-bar series
(the regime signal is completely blind) raises `GateFeederError`; a merely
*short* history degrades to `None` for the vol inputs, which `gate_state`
skips safely -- the event-calendar / OPEX / injected-VIX triggers still
apply.

Tests: `vt/tests/test_gate_feeder.py`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

from vt.data.feed import Bar
from vt.gate.calendar import GateState, gate_state
from vt.pipeline.runner import FeedProtocol

DEFAULT_REALIZED_VOL_WINDOW = 10  # trading days in each rolling-stdev sample
DEFAULT_REALIZED_VOL_HISTORY_LEN = 60  # Strategy_Spec.md Sec2 trailing-60d base
DEFAULT_INDEX_RETURNS_WINDOW = 21  # ~1 month for the 2-sigma move check
DEFAULT_DAILY_LOOKBACK = 120  # daily bars to pull (covers 60 + window + slack)

_EQUITY_INDEX = "SPY"
_CRYPTO_INDEX = "BTC-USDT"
_CRYPTO_VENUE = "okx"


class GateFeederError(RuntimeError):
    """Raised when the regime gate cannot be fed at all -- e.g. no index
    daily bars came back, so there is no basis to assess the session
    regime. A short-but-nonempty history is not an error (it degrades to
    skipped vol triggers), only a total absence is.
    """


def build_gate_state(
    feed: FeedProtocol,
    asof: datetime,
    *,
    venue: str = "alpaca",
    index_symbol: str | None = None,
    vix: float | None = None,
    realized_vol_window: int = DEFAULT_REALIZED_VOL_WINDOW,
    realized_vol_history_len: int = DEFAULT_REALIZED_VOL_HISTORY_LEN,
    index_returns_window: int = DEFAULT_INDEX_RETURNS_WINDOW,
    daily_lookback: int = DEFAULT_DAILY_LOOKBACK,
    events_path: Path | None = None,
) -> GateState:
    """Resolve the session `GateState` from live index data.

    Parameters
    ----------
    feed:
        The same `FeedProtocol` the pipeline uses (`vt.data.feed` in
        production, a fake in tests). Used to pull the index's daily bars.
    asof:
        Session timestamp; forwarded to `gate_state` (calendar-date
        triggers) and used as the `end` of the daily-bar pull.
    venue:
        Selects the default index proxy when `index_symbol` is not given:
        SPY for equities, BTC-USDT for crypto.
    index_symbol:
        Explicit index symbol override (e.g. a different benchmark).
    vix:
        Optional current VIX level, injected. `None` (the default) means
        the VIX trigger is simply not evaluated -- see the module docstring
        for why there is no built-in VIX fetch yet.
    realized_vol_window, realized_vol_history_len, index_returns_window,
    daily_lookback:
        Windowing knobs; defaults follow Strategy_Spec.md Sec2.
    events_path:
        Override the bundled events YAML (forwarded to `gate_state`).

    Raises
    ------
    GateFeederError:
        The index returned zero daily bars.
    """
    symbol = index_symbol or (_CRYPTO_INDEX if venue == _CRYPTO_VENUE else _EQUITY_INDEX)
    bars = feed.get_bars(symbol, "1d", end=asof, limit=daily_lookback)
    if not bars:
        raise GateFeederError(
            f"no daily bars for index {symbol!r} -- cannot assess session regime"
        )

    ordered = sorted(bars, key=lambda b: b.time)
    returns = _daily_returns(ordered)

    realized_vol, realized_vol_history = _realized_vol_inputs(
        returns, window=realized_vol_window, history_len=realized_vol_history_len
    )
    index_returns = returns[-index_returns_window:] if len(returns) >= 2 else None

    return gate_state(
        asof,
        vix=vix,
        realized_vol=realized_vol,
        realized_vol_history=realized_vol_history,
        index_returns=index_returns,
        events_path=events_path,
    )


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def _daily_returns(ordered: Sequence[Bar]) -> list[float]:
    """Close-to-close daily returns as decimal fractions. Bars with a
    non-positive prior close are skipped (a ratio against them is
    meaningless)."""
    out: list[float] = []
    for i in range(1, len(ordered)):
        prev = ordered[i - 1].close
        if prev <= 0:
            continue
        out.append(ordered[i].close / prev - 1.0)
    return out


def _stdev(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return var**0.5


def _rolling_stdev(returns: Sequence[float], window: int) -> list[float]:
    """One sample-stdev per full trailing `window` of returns; index -1 is
    the most recent window. Empty when there aren't `window` returns."""
    if window < 2 or len(returns) < window:
        return []
    return [_stdev(returns[end - window : end]) for end in range(window, len(returns) + 1)]


def _realized_vol_inputs(
    returns: Sequence[float], *, window: int, history_len: int
) -> tuple[float | None, list[float] | None]:
    """(today's realized vol, trailing history) for `gate_state`.

    `realized_vol` is the most recent rolling-stdev sample; the history is
    the up-to-`history_len` samples *before* it (today excluded, so the
    percentile threshold isn't biased by the value being tested against
    it). Both `None` when there isn't even one full window of returns --
    `gate_state` then skips the realized-vol trigger.
    """
    vols = _rolling_stdev(returns, window)
    if not vols:
        return None, None
    realized_vol = vols[-1]
    prior = vols[:-1][-history_len:]
    return realized_vol, (prior or None)
