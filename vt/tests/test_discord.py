"""Tests for M012 -- vt.alerts.discord (Phase D, VT037).

Verifies the alerter's invariants without ever calling Discord:
  * payload shape (content, allowed_mentions)
  * mention formatting (`<@ID>` prepended, allowed_mentions.users set)
  * fail-open behaviour: missing config, HTTP error, non-2xx status
    all return False without raising
  * Discord's 2000-char message limit enforced here, not at the wire
  * config file parsing (present / absent / partially populated)
  * semantic helpers (breaker_trip, drift, kill_switch) all mention
    the user (these are the events that need to page them)

Run with: pytest vt/tests -m unit
"""

from __future__ import annotations

import json
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pytest

from vt.alerts import discord

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fake HTTP sink -- records every POST without touching the network
# --------------------------------------------------------------------------- #


@dataclass
class FakeSink:
    status: int = 204  # Discord webhooks return 204 No Content on success
    raise_error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, url: str, *, payload: Mapping[str, Any], timeout: float) -> int:
        self.calls.append({"url": url, "payload": dict(payload), "timeout": timeout})
        if self.raise_error is not None:
            raise self.raise_error
        return self.status


def _config(**overrides) -> discord.DiscordConfig:
    return discord.DiscordConfig(
        webhook_url=overrides.get(
            "webhook_url", "https://discord.com/api/webhooks/1234/token"
        ),
        mention_user_id=overrides.get("mention_user_id", "123456789012345678"),
    )


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #


def test_load_config_returns_empty_when_file_missing(tmp_path: Path) -> None:
    cfg = discord.load_config(tmp_path / "no_such_file.json")
    assert cfg.webhook_url is None
    assert cfg.mention_user_id is None
    assert discord.is_configured(cfg) is False


def test_load_config_reads_both_fields(tmp_path: Path) -> None:
    p = tmp_path / "discord.json"
    p.write_text(
        json.dumps(
            {
                "webhook_url": "https://discord.com/api/webhooks/1/tok",
                "mention_user_id": "123456789012345678",
            }
        ),
        encoding="utf-8",
    )
    cfg = discord.load_config(p)
    assert cfg.webhook_url == "https://discord.com/api/webhooks/1/tok"
    assert cfg.mention_user_id == "123456789012345678"
    assert discord.is_configured(cfg) is True


def test_load_config_coerces_numeric_mention_id_to_str(tmp_path: Path) -> None:
    """Discord user IDs are 64-bit snowflakes. If someone writes them
    as a JSON number (no quotes) they'd overflow round-tripping through
    some parsers; guard by coercing to str at load time."""
    p = tmp_path / "discord.json"
    p.write_text(
        json.dumps({"webhook_url": "https://x", "mention_user_id": 123456789012345678}),
        encoding="utf-8",
    )
    cfg = discord.load_config(p)
    assert cfg.mention_user_id == "123456789012345678"
    assert isinstance(cfg.mention_user_id, str)


def test_load_config_treats_empty_webhook_as_unconfigured(tmp_path: Path) -> None:
    """The whole point of the None-webhook path is to let the file
    exist with just the mention_user_id filled in while the user is
    still fetching the webhook URL from Discord."""
    p = tmp_path / "discord.json"
    p.write_text(
        json.dumps({"webhook_url": "", "mention_user_id": "123456789012345678"}),
        encoding="utf-8",
    )
    cfg = discord.load_config(p)
    assert cfg.webhook_url is None
    assert discord.is_configured(cfg) is False


def test_load_config_raises_on_non_object_root(tmp_path: Path) -> None:
    p = tmp_path / "discord.json"
    p.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON object"):
        discord.load_config(p)


# --------------------------------------------------------------------------- #
# alert() -- happy path, mention formatting, allowed_mentions
# --------------------------------------------------------------------------- #


def test_alert_posts_content_and_returns_true_on_success() -> None:
    sink = FakeSink()
    alerter = discord.DiscordAlerter(config=_config(), http=sink)

    assert alerter.alert("system booted") is True
    assert len(sink.calls) == 1
    call = sink.calls[0]
    assert call["url"] == "https://discord.com/api/webhooks/1234/token"
    assert call["payload"]["content"] == "system booted"
    # No mention requested -> no allowed_mentions payload.
    assert "allowed_mentions" not in call["payload"]


def test_alert_with_mention_prepends_user_tag() -> None:
    sink = FakeSink()
    alerter = discord.DiscordAlerter(config=_config(), http=sink)

    alerter.alert("check this", mention=True)
    payload = sink.calls[0]["payload"]
    assert payload["content"] == "<@123456789012345678> check this"


def test_alert_with_mention_sets_allowed_mentions_so_the_user_actually_pings() -> None:
    """Discord webhooks default to NOT pinging even when the message
    body contains a <@ID> token. The allowed_mentions payload is what
    turns the mention from decorative-text into an actual notification
    -- exactly what a breaker-trip alert needs."""
    sink = FakeSink()
    alerter = discord.DiscordAlerter(config=_config(), http=sink)

    alerter.alert("wake up", mention=True)
    payload = sink.calls[0]["payload"]
    assert payload["allowed_mentions"] == {"users": ["123456789012345678"]}


def test_alert_without_mention_id_still_sends_but_does_not_prepend() -> None:
    """Config with webhook but no mention_user_id: alerts still send
    (webhook is what matters for delivery), just no @ping."""
    sink = FakeSink()
    alerter = discord.DiscordAlerter(
        config=_config(mention_user_id=None), http=sink
    )

    assert alerter.alert("hello", mention=True) is True
    payload = sink.calls[0]["payload"]
    assert payload["content"] == "hello"  # no <@None> garbage
    assert "allowed_mentions" not in payload


# --------------------------------------------------------------------------- #
# Fail-open -- missing config / HTTP failure / non-2xx never raise
# --------------------------------------------------------------------------- #


def test_alert_returns_false_and_does_not_call_http_when_unconfigured() -> None:
    sink = FakeSink()
    alerter = discord.DiscordAlerter(
        config=discord.DiscordConfig(webhook_url=None, mention_user_id=None), http=sink
    )
    assert alerter.alert("nothing to see") is False
    assert sink.calls == []


def test_alert_returns_false_on_http_transport_error() -> None:
    sink = FakeSink(raise_error=urllib.error.URLError("connection refused"))
    alerter = discord.DiscordAlerter(config=_config(), http=sink)
    assert alerter.alert("boom") is False


def test_alert_returns_false_on_timeout_error() -> None:
    sink = FakeSink(raise_error=TimeoutError("read timed out"))
    alerter = discord.DiscordAlerter(config=_config(), http=sink)
    assert alerter.alert("slow") is False


def test_alert_returns_false_on_non_2xx_response() -> None:
    sink = FakeSink(status=429)  # Discord rate limit
    alerter = discord.DiscordAlerter(config=_config(), http=sink)
    assert alerter.alert("slow down") is False


def test_alert_does_not_raise_on_generic_os_error() -> None:
    """An OSError (DNS failure, socket exhaustion) must never surface
    to the caller -- the fail-open invariant matters most when the
    network is unhealthy, which is exactly when this branch fires."""
    sink = FakeSink(raise_error=OSError("network down"))
    alerter = discord.DiscordAlerter(config=_config(), http=sink)
    assert alerter.alert("test") is False


# --------------------------------------------------------------------------- #
# Message length -- Discord 2000-char cap enforced here
# --------------------------------------------------------------------------- #


def test_long_message_is_truncated_below_discord_limit() -> None:
    sink = FakeSink()
    alerter = discord.DiscordAlerter(config=_config(), http=sink)
    huge = "x" * 5000

    alerter.alert(huge)
    content = sink.calls[0]["payload"]["content"]
    assert len(content) <= discord.DISCORD_MESSAGE_LIMIT
    assert content.endswith("…[truncated]")


def test_short_message_is_not_touched() -> None:
    sink = FakeSink()
    alerter = discord.DiscordAlerter(config=_config(), http=sink)
    alerter.alert("brief")
    assert sink.calls[0]["payload"]["content"] == "brief"


# --------------------------------------------------------------------------- #
# Semantic helpers -- three events that MUST @mention the user
# --------------------------------------------------------------------------- #


def test_alert_breaker_trip_mentions_user() -> None:
    sink = FakeSink()
    alerter = discord.DiscordAlerter(config=_config(), http=sink)

    alerter.alert_breaker_trip(breaker="daily_loss_limit", detail="session_r=-2.1")
    payload = sink.calls[0]["payload"]
    assert "<@123456789012345678>" in payload["content"]
    assert "daily_loss_limit" in payload["content"]
    assert "session_r=-2.1" in payload["content"]
    assert payload["allowed_mentions"] == {"users": ["123456789012345678"]}


def test_alert_drift_mentions_user_and_reports_both_quantities() -> None:
    sink = FakeSink()
    alerter = discord.DiscordAlerter(config=_config(), http=sink)

    alerter.alert_drift(venue="alpaca", symbol="AAPL", internal_qty=13.0, broker_qty=26.0)
    payload = sink.calls[0]["payload"]
    assert "<@123456789012345678>" in payload["content"]
    assert "alpaca" in payload["content"]
    assert "AAPL" in payload["content"]
    assert "13" in payload["content"]
    assert "26" in payload["content"]
    assert payload["allowed_mentions"] == {"users": ["123456789012345678"]}


def test_alert_kill_switch_reports_counts_and_error_status() -> None:
    sink = FakeSink()
    alerter = discord.DiscordAlerter(config=_config(), http=sink)

    alerter.alert_kill_switch(cancelled=3, closed=2, had_errors=False)
    clean_payload = sink.calls[0]["payload"]
    assert "clean" in clean_payload["content"]
    assert "cancelled: 3" in clean_payload["content"]
    assert "closed: 2" in clean_payload["content"]

    alerter.alert_kill_switch(cancelled=1, closed=0, had_errors=True)
    error_payload = sink.calls[1]["payload"]
    assert "with errors" in error_payload["content"]


# --------------------------------------------------------------------------- #
# default_alerter factory
# --------------------------------------------------------------------------- #


def test_default_alerter_survives_missing_credentials_file(tmp_path: Path) -> None:
    """Production code paths must be able to construct an alerter
    without knowing whether the user has set up Discord yet. Missing
    file -> unconfigured alerter that no-ops on alert(), not a crash."""
    alerter = discord.default_alerter(credentials_path=tmp_path / "nope.json")
    assert isinstance(alerter, discord.DiscordAlerter)
    assert discord.is_configured(alerter.config) is False
    # And its alert() does not raise or POST anywhere.
    assert alerter.alert("test") is False
