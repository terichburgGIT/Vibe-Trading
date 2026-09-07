"""Tests for M012 -- vt.alerts.kill (T023; see 06_Tests.md).

Phase C4 (`16_Next_Steps.md`): written against a STUB where every
`vt.alerts.kill` function raises NotImplementedError. Every test in this
file must fail with NotImplementedError before the real implementation
lands. Spec sources: `03_Modules.md` M012, `Strategy_Spec.md` /
`Risk_Policy.md` (kill-switch section), `16_Next_Steps.md` C4.

T023 as specified says "SIGSTOP the main process, run the standalone
kill script." SIGSTOP is Unix-only; on Windows the invariant we actually
care about is the *structural* one -- the kill script must not depend on
any main-app state -- plus the *behavioural* one -- from a clean
subprocess with only credentials on disk it must be able to cancel
orders and flatten positions on the broker. Both are covered here,
Windows/Linux/macOS portable.

Run with: pytest vt/tests -m unit
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from vt.alerts import kill

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fake adapter -- used to exercise orchestration without hitting any broker
# --------------------------------------------------------------------------- #


@dataclass
class FakeAdapter:
    venue: str
    open_orders: list[str] = field(default_factory=list)
    positions: list[str] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    closed: list[str] = field(default_factory=list)
    fail_cancel: set[str] = field(default_factory=set)
    fail_close: set[str] = field(default_factory=set)

    def list_open_orders(self) -> list[str]:
        return list(self.open_orders)

    def list_positions(self) -> list[str]:
        return list(self.positions)

    def cancel_order(self, order_id: str) -> None:
        if order_id in self.fail_cancel:
            raise RuntimeError(f"cancel {order_id} failed")
        self.cancelled.append(order_id)

    def close_position(self, symbol: str) -> None:
        if symbol in self.fail_close:
            raise RuntimeError(f"close {symbol} failed")
        self.closed.append(symbol)


# --------------------------------------------------------------------------- #
# T023 -- orchestration behaviour
# --------------------------------------------------------------------------- #


def test_kill_all_cancels_every_order_and_closes_every_position() -> None:
    a = FakeAdapter(
        venue="alpaca",
        open_orders=["ord-1", "ord-2"],
        positions=["AAPL", "MSFT"],
    )
    report = kill.kill_all([a])

    assert report.dry_run is False
    assert a.cancelled == ["ord-1", "ord-2"]
    assert a.closed == ["AAPL", "MSFT"]
    assert report.total_cancelled == 2
    assert report.total_closed == 2
    assert report.had_errors is False


def test_kill_all_cancels_orders_before_closing_positions() -> None:
    """A resting stop must not fire because the position was closed first."""
    call_order: list[str] = []

    class OrderTracker(FakeAdapter):
        def cancel_order(self, order_id: str) -> None:
            call_order.append(f"cancel:{order_id}")
            super().cancel_order(order_id)

        def close_position(self, symbol: str) -> None:
            call_order.append(f"close:{symbol}")
            super().close_position(symbol)

    a = OrderTracker(
        venue="alpaca",
        open_orders=["ord-1"],
        positions=["AAPL"],
    )
    kill.kill_all([a])

    assert call_order == ["cancel:ord-1", "close:AAPL"]


def test_kill_all_dry_run_makes_no_side_effect_calls() -> None:
    a = FakeAdapter(
        venue="alpaca",
        open_orders=["ord-1"],
        positions=["AAPL"],
    )
    report = kill.kill_all([a], dry_run=True)

    assert report.dry_run is True
    assert a.cancelled == []
    assert a.closed == []
    # But the report still names what *would* be acted on:
    assert report.total_cancelled == 1
    assert report.total_closed == 1


def test_kill_all_continues_across_venues_on_failure() -> None:
    """One venue erroring must not stop the other from being flattened."""
    a = FakeAdapter(
        venue="alpaca",
        open_orders=["ord-1"],
        positions=["AAPL"],
        fail_close={"AAPL"},
    )
    b = FakeAdapter(
        venue="okx",
        open_orders=["okx-ord-1"],
        positions=["BTC-USDT"],
    )

    report = kill.kill_all([a, b])

    # b was fully flattened even though a errored.
    assert b.cancelled == ["okx-ord-1"]
    assert b.closed == ["BTC-USDT"]
    assert report.had_errors is True
    errors_a = next(r for r in report.results if r.venue == "alpaca").errors
    errors_b = next(r for r in report.results if r.venue == "okx").errors
    assert len(errors_a) >= 1
    assert errors_b == ()


def test_kill_all_continues_across_orders_on_single_cancel_failure() -> None:
    """A failing cancel_order must not skip the remaining orders."""
    a = FakeAdapter(
        venue="alpaca",
        open_orders=["ord-1", "ord-2", "ord-3"],
        positions=[],
        fail_cancel={"ord-2"},
    )

    report = kill.kill_all([a])

    assert a.cancelled == ["ord-1", "ord-3"]
    assert report.had_errors is True


def test_kill_all_with_zero_adapters_is_a_no_op() -> None:
    report = kill.kill_all([])
    assert report.dry_run is False
    assert report.results == ()
    assert report.total_cancelled == 0
    assert report.total_closed == 0
    assert report.had_errors is False


# --------------------------------------------------------------------------- #
# Structural invariants -- the "no shared state" spec requirement
# --------------------------------------------------------------------------- #


def test_kill_module_does_not_import_from_main_app() -> None:
    """T023 spec: no dependency on main-process state. Enforced structurally.

    The kill script must not import from any other `vt.*` submodule that
    carries live app state (data feed, risk gate, universe, execution,
    journal, indicators, etc.). Stdlib, third-party broker SDKs, and yaml
    are fine. Absolutely no `agent.*` import either -- that whole tree is
    the wrapped upstream app and would defeat the point.
    """
    source = Path(kill.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    forbidden_prefixes = (
        "agent.",
        "vt.data",
        "vt.risk",
        "vt.universe",
        "vt.exec",
        "vt.indicators",
        "vt.journal",
        "vt.render",
        "vt.signal",
        "vt.analyst",
        "vt.validate",
        "vt.gate",
    )

    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    for name in imported:
        assert not name.startswith(forbidden_prefixes), (
            f"kill.py imports {name!r} -- would create shared state with the "
            "wedged main app, defeating the whole point of a kill switch."
        )


def test_kill_module_can_be_imported_without_broker_sdks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing `vt.alerts.kill` must not require alpaca-py / python-okx.

    Reason: on a wedged host you may need to run the kill script from a
    minimal Python that has only the deps for the venue you're targeting.
    SDK imports must happen lazily inside the concrete adapter loaders.
    """
    import importlib

    # Simulate alpaca-py + python-okx being absent by forcing ImportError.
    import builtins

    real_import = builtins.__import__

    def _guarded(name: str, *args, **kwargs):
        if name.startswith(("alpaca", "okx")):
            raise ImportError(f"pretend {name} isn't installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded)

    # Reimporting the module under the guard must succeed.
    import vt.alerts.kill as km  # noqa: F401

    importlib.reload(km)


# --------------------------------------------------------------------------- #
# Credentials path -- "own credentials path" spec requirement
# --------------------------------------------------------------------------- #


def test_alpaca_loader_reads_from_override_credentials_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller must be able to point the loader at a custom creds dir."""
    creds = {
        "key_id": "PKTESTKEY",
        "secret_key": "sk-test-XXXX",
        "is_paper": True,
    }
    (tmp_path / "alpaca.json").write_text(json.dumps(creds), encoding="utf-8")

    # Stub out alpaca-py TradingClient so the loader doesn't try to reach
    # the network. What we care about here is only that our creds path was
    # honored and the client was constructed with those key/secret values.
    captured: dict[str, object] = {}

    class FakeTradingClient:
        def __init__(self, api_key: str, secret_key: str, paper: bool = True) -> None:
            captured["api_key"] = api_key
            captured["secret_key"] = secret_key
            captured["paper"] = paper

    import types

    fake_alpaca = types.ModuleType("alpaca")
    fake_trading = types.ModuleType("alpaca.trading")
    fake_client_mod = types.ModuleType("alpaca.trading.client")
    fake_client_mod.TradingClient = FakeTradingClient
    fake_alpaca.trading = fake_trading
    fake_trading.client = fake_client_mod

    fake_requests_mod = types.ModuleType("alpaca.trading.requests")

    class _Req:
        def __init__(self, *a, **kw) -> None:
            pass

    fake_requests_mod.GetOrdersRequest = _Req
    fake_requests_mod.ClosePositionRequest = _Req

    fake_enums_mod = types.ModuleType("alpaca.trading.enums")
    fake_enums_mod.QueryOrderStatus = types.SimpleNamespace(OPEN="open")

    monkeypatch.setitem(sys.modules, "alpaca", fake_alpaca)
    monkeypatch.setitem(sys.modules, "alpaca.trading", fake_trading)
    monkeypatch.setitem(sys.modules, "alpaca.trading.client", fake_client_mod)
    monkeypatch.setitem(sys.modules, "alpaca.trading.requests", fake_requests_mod)
    monkeypatch.setitem(sys.modules, "alpaca.trading.enums", fake_enums_mod)

    adapter = kill.load_alpaca_adapter(credentials_dir=tmp_path)

    assert adapter.venue == "alpaca"
    assert captured["api_key"] == "PKTESTKEY"
    assert captured["secret_key"] == "sk-test-XXXX"
    assert captured["paper"] is True


def test_alpaca_loader_raises_clearly_when_creds_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        kill.load_alpaca_adapter(credentials_dir=tmp_path)


def test_default_credentials_dir_matches_project_convention() -> None:
    """Same convention as `~/.vibe-trading/*.json` used elsewhere."""
    assert kill.DEFAULT_CREDENTIALS_DIR.name == ".vibe-trading"
    assert kill.DEFAULT_CREDENTIALS_DIR.parent == Path.home()


# --------------------------------------------------------------------------- #
# T023 -- CLI runs standalone from a fresh subprocess
# --------------------------------------------------------------------------- #


def test_cli_runs_standalone_from_clean_subprocess(tmp_path: Path) -> None:
    """Windows-portable equivalent of "SIGSTOP main, then run the script".

    A brand-new Python subprocess with only creds on disk must be able to
    boot the module and execute the kill flow, without importing from the
    main app or needing the main app to be reachable. This is the
    invariant T023 is really after: kill runs when everything else is
    frozen.

    We invoke `python -m vt.alerts.kill --dry-run --no-venues` -- dry-run
    means no broker calls; `--no-venues` means no adapters loaded, so no
    real credentials are needed. Exit 0 + a report on stdout means the
    entry point is intact end-to-end from a fresh process.
    """
    result = subprocess.run(
        [sys.executable, "-m", "vt.alerts.kill", "--dry-run", "--no-venues"],
        cwd=Path(__file__).resolve().parents[2],  # Source/
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "dry_run" in result.stdout.lower()


def test_main_returns_zero_on_success_with_no_venues() -> None:
    assert kill.main(["--dry-run", "--no-venues"]) == 0


def test_main_returns_nonzero_when_any_venue_errors() -> None:
    """CLI exit code must surface partial failures so ops sees them."""

    def _factory() -> kill.BrokerKillAdapter:
        return FakeAdapter(
            venue="alpaca",
            open_orders=["ord-1"],
            positions=["AAPL"],
            fail_close={"AAPL"},
        )

    rc = kill.main(["--dry-run", "--no-venues"], _extra_adapters=[_factory()])  # type: ignore[call-arg]
    # dry-run does not attempt the failing close, so this exits clean --
    # the failure only manifests on a real run:
    assert rc == 0

    rc = kill.main([], _extra_adapters=[_factory()])  # type: ignore[call-arg]
    assert rc != 0
