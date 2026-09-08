"""M012 -- Discord alerting. Phase D, VT037.

Fires a Discord webhook when the risk spine trips a breaker, when
reconciliation reports drift, or when the kill switch runs. Small
wrapper over `requests`-less HTTP: uses stdlib `urllib` so this module
adds no new dep to the critical path (same reasoning as JSONL over
Parquet in `vt/journal/store.py`).

Design invariants:
  * **Config-driven, never hard-coded**. Webhook URL and mention target
    live in `~/.vibe-trading/discord.json` -- same convention as the
    Alpaca/OKX credentials. Missing config is NOT a crash: `alert()`
    logs a warning and returns False so a breaker trip is never lost
    because the alerter wasn't configured yet.
  * **Fail-open, never fail-closed on alert send**. If the HTTP call
    to Discord raises or times out, we swallow the error and return
    False -- the caller has more important things to do (e.g. actually
    flatten a position) than to be blocked on an alerter timeout.
  * **Discord's 2000-char message limit is enforced here**, not left
    for Discord to reject silently. Long payloads are truncated with a
    "[truncated]" tail that stays inside the limit.
  * **@mentions require Discord's allowed_mentions payload** to
    actually notify the user (webhooks default to no ping even when
    the message contains a `<@ID>` token). We set `allowed_mentions`
    explicitly so a real trip actually pages the user.

Config file format (`~/.vibe-trading/discord.json`)::

    {
      "webhook_url": "https://discord.com/api/webhooks/<id>/<token>",
      "mention_user_id": "<your Discord user ID snowflake, as a string>"
    }

Full contract in `03_Modules.md` section M012.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol


log = logging.getLogger(__name__)

DEFAULT_CREDENTIALS_DIR = Path.home() / ".vibe-trading"
DISCORD_MESSAGE_LIMIT = 2000  # Discord API hard cap
DEFAULT_HTTP_TIMEOUT_SECONDS = 5.0
_TRUNCATION_TAIL = "\n…[truncated]"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DiscordConfig:
    """Webhook target + who to @mention on a trip. `webhook_url` is
    None-able so `load_config` can return a valid object even before
    the user has pasted the webhook URL into the file -- the alerter
    then no-ops with a logged warning instead of crashing every code
    path that tries to send a notification.
    """

    webhook_url: str | None
    mention_user_id: str | None


def load_config(path: Path | None = None) -> DiscordConfig:
    """Read `~/.vibe-trading/discord.json` (or an override path).
    Missing file returns an empty config -- callers should check
    `is_configured()` before assuming a send will happen.
    """
    creds_path = path or (DEFAULT_CREDENTIALS_DIR / "discord.json")
    if not creds_path.exists():
        return DiscordConfig(webhook_url=None, mention_user_id=None)
    with creds_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{creds_path}: expected a JSON object, got {type(raw).__name__}")
    return DiscordConfig(
        webhook_url=raw.get("webhook_url") or None,
        mention_user_id=(str(raw["mention_user_id"]) if raw.get("mention_user_id") else None),
    )


def is_configured(config: DiscordConfig) -> bool:
    """True iff the alerter can actually POST to Discord. A config with
    only a `mention_user_id` and no `webhook_url` is NOT configured --
    a mention with nowhere to send it is a no-op.
    """
    return bool(config.webhook_url)


# --------------------------------------------------------------------------- #
# HTTP sink -- Protocol so tests can inject a fake without patching urllib
# --------------------------------------------------------------------------- #


class HttpSink(Protocol):
    def __call__(self, url: str, *, payload: Mapping[str, Any], timeout: float) -> int: ...


def _urllib_post(url: str, *, payload: Mapping[str, Any], timeout: float) -> int:
    """Default HTTP sink -- POST JSON via stdlib urllib. Returns the
    HTTP status code. Raises urllib.error.URLError on transport
    failure; the alerter wraps this in a try/except so callers never
    see the exception.
    """
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 -- URL is user-configured
        return resp.status


# --------------------------------------------------------------------------- #
# Alerter
# --------------------------------------------------------------------------- #


@dataclass
class DiscordAlerter:
    """Small, testable object. In production, construct with default
    config load and default HTTP sink. In tests, inject a FakeSink and
    a DiscordConfig with an arbitrary webhook_url to record what would
    have been posted.
    """

    config: DiscordConfig
    http: HttpSink = _urllib_post
    timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS

    def alert(self, message: str, *, mention: bool = False) -> bool:
        """Send one Discord message. Returns True iff the POST returned
        a 2xx status. Returns False (never raises) on:
          * missing/incomplete config,
          * HTTP transport error,
          * non-2xx response.
        Callers may log the False but must not treat it as a fatal
        error -- a breaker trip is more important than a Discord ping.
        """
        if not is_configured(self.config):
            log.warning("discord alerter not configured; dropping message: %s", message[:80])
            return False

        content = _prepend_mention(message, self.config.mention_user_id) if mention else message
        content = _truncate(content, DISCORD_MESSAGE_LIMIT)

        payload: dict[str, Any] = {"content": content}
        if mention and self.config.mention_user_id:
            # Without this, Discord webhooks render the <@ID> as text
            # but do not send a notification -- exactly the opposite of
            # what a breaker-trip alert needs.
            payload["allowed_mentions"] = {"users": [self.config.mention_user_id]}

        try:
            status = self.http(self.config.webhook_url, payload=payload, timeout=self.timeout_seconds)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            log.warning("discord alerter HTTP failure (%s); message dropped", exc)
            return False
        if not (200 <= status < 300):
            log.warning("discord alerter got HTTP %s; message dropped", status)
            return False
        return True

    # ------------------------------------------------------------------ #
    # Semantic helpers -- the caller shouldn't have to format the message
    # the same way every time; these bake in the shape and the mention
    # decision (all three "attention required" events mention the user).
    # ------------------------------------------------------------------ #

    def alert_breaker_trip(self, *, breaker: str, detail: str) -> bool:
        return self.alert(
            f"🛑 **Breaker trip: {breaker}**\n{detail}",
            mention=True,
        )

    def alert_drift(self, *, venue: str, symbol: str, internal_qty: float, broker_qty: float) -> bool:
        return self.alert(
            f"⚠️ **Reconciliation drift halt**\n"
            f"venue=`{venue}` symbol=`{symbol}` "
            f"internal={internal_qty} broker={broker_qty}\n"
            f"No further orders until manual clear.",
            mention=True,
        )

    def alert_kill_switch(self, *, cancelled: int, closed: int, had_errors: bool) -> bool:
        icon = "🚨" if had_errors else "🧯"
        status = "with errors" if had_errors else "clean"
        return self.alert(
            f"{icon} **Kill switch fired ({status})**\n"
            f"orders cancelled: {cancelled} · positions closed: {closed}",
            mention=True,
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _prepend_mention(message: str, user_id: str | None) -> str:
    if not user_id:
        return message
    return f"<@{user_id}> {message}"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    keep = limit - len(_TRUNCATION_TAIL)
    if keep <= 0:  # pathological -- limit smaller than the tail itself
        return text[:limit]
    return text[:keep] + _TRUNCATION_TAIL


# --------------------------------------------------------------------------- #
# Convenience factory -- what production callers use
# --------------------------------------------------------------------------- #


def default_alerter(
    *,
    credentials_path: Path | None = None,
    http: HttpSink | None = None,
) -> DiscordAlerter:
    """Build an alerter from `~/.vibe-trading/discord.json`. Never
    raises on a missing/unconfigured file -- the returned alerter's
    `alert()` will simply no-op with a warning until config lands.
    """
    return DiscordAlerter(
        config=load_config(credentials_path),
        http=http or _urllib_post,
    )
