"""Tests for qf.cli -- argument wiring, exit codes, and that refusals come
back as readable messages rather than tracebacks.

Run with: pytest qf/tests -m unit
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from qf.cli import main
from qf.tests.conftest import SYM, _bars

pytestmark = pytest.mark.unit


def test_plan_prints_one_line_per_symbol_and_places_no_orders(rig, capsys):
    qf, broker, _ = rig
    assert main(["plan", SYM, "ETH-USDT", "--lane", "large"], engine=qf) == 0
    out = capsys.readouterr().out
    assert out.count("PASS") == 2 and "large $4,000" in out and "no orders placed" in out
    assert broker.calls == []


def test_plan_flags_below_floor(rig, capsys):
    qf, _, _ = rig
    qf._get_bars = lambda s, t, limit: _bars(s, t, limit=limit, rng=1.0)
    main(["plan", SYM], engine=qf)
    assert "BELOW FEE FLOOR" in capsys.readouterr().out


def test_open_then_status_then_run_once(rig, capsys):
    qf, broker, clock = rig
    assert main(["open", SYM, "--lane", "small", "--manual"], engine=qf) == 0
    assert "opened qfSOL" in capsys.readouterr().out

    main(["status"], engine=qf)
    out = capsys.readouterr().out
    assert "halted: False" in out and "qfSOL" in out and "closed manual: n=0" in out

    pos = qf.book().positions[0]
    broker.trigger(pos.algo_id, "tp", 110.0, pos.size)
    assert main(["run", "--once"], engine=qf) == 0
    out = capsys.readouterr().out
    assert "target" in out and "no open positions" in out


def test_refusal_is_a_message_and_exit_code_2(rig, capsys):
    qf, _, _ = rig
    assert main(["open", SYM, "--lane", "small"], engine=qf) == 2  # no --manual
    assert "refused:" in capsys.readouterr().err


def test_unknown_lane_is_an_argparse_error(rig, capsys):
    qf, _, _ = rig
    with pytest.raises(SystemExit):
        main(["open", SYM, "--lane", "huge", "--manual"], engine=qf)
    assert "invalid choice" in capsys.readouterr().err


def test_plan_error_is_refused_readably(rig, capsys):
    qf, _, _ = rig
    qf._get_bars = lambda s, t, limit: _bars(s, t, limit=5)
    assert main(["open", SYM, "--lane", "small", "--manual"], engine=qf) == 2
    assert "refused:" in capsys.readouterr().err


def test_kill_dry_run_and_resume(rig, capsys):
    qf, _, clock = rig
    main(["open", SYM, "--lane", "small", "--manual"], engine=qf)
    assert main(["kill", "--dry-run"], engine=qf) == 0
    assert "would close" in capsys.readouterr().out
    assert qf.book().positions  # untouched

    assert main(["kill"], engine=qf) == 0
    assert qf.book().halted
    assert main(["resume"], engine=qf) == 0
    assert not qf.book().halted


def test_run_once_reports_halt_in_exit_code(rig):
    qf, broker, clock = rig
    main(["open", SYM, "--lane", "small", "--manual"], engine=qf)
    clock.now += timedelta(hours=9)
    broker.fail.add("market_sell")
    assert main(["run", "--once"], engine=qf) == 1
