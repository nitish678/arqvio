"""Audited alerts (DR §6): the sha256 chain and its tamper detection, audit-before-delivery, the adapters'
contract (SENT / FAILED / SKIPPED, never raising, secrets never in a detail), the schema, the call-time spool path,
and the drift-fire -> audited-alert end-to-end fixture."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.research.alerts.adapters import DesktopAdapter, EmailAdapter, TelegramAdapter
from engine.research.alerts.dispatch import emit
from engine.research.alerts.producers import from_drift, from_replay
from engine.research.alerts.schema import AlertSchemaError, make_alert
from engine.research.alerts.spool import GENESIS, spool_path, tail, verify

CLOCK = lambda: "2026-10-02T00:00:00+00:00"  # noqa: E731 -- an injected, fixed clock


def _alert(kind: str = "HEAT_HIGH", **payload) -> dict:
    return make_alert(kind, "warn", payload or {"x": 1}, "artifacts/portfolio/replay.json", clock=CLOCK)


def _spool(tmp_path: Path, n: int = 3) -> Path:
    p = tmp_path / "spool.jsonl"
    for k in range(n):
        emit(_alert(x=k), path=p, clock=CLOCK)
    return p


def test_the_spool_chains_from_genesis(tmp_path: Path) -> None:
    p = _spool(tmp_path)
    rows = [json.loads(r) for r in p.read_text(encoding="ascii").splitlines()]
    assert [r["seq"] for r in rows] == [0, 1, 2] and rows[0]["prev_hash"] == GENESIS
    assert all(rows[k]["prev_hash"] == rows[k - 1]["line_hash"] for k in (1, 2))
    assert verify(p).ok and verify(p).lines == 3


@pytest.mark.parametrize("mode", ["edit_payload", "edit_hash", "drop_line", "swap_lines"])
def test_any_tamper_is_detected_at_its_line(tmp_path: Path, mode: str) -> None:
    p = _spool(tmp_path, 4)
    lines = p.read_text(encoding="ascii").splitlines()
    if mode == "edit_payload":
        lines[1] = lines[1].replace('"x":1', '"x":7')
        want = 2
    elif mode == "edit_hash":
        lines[2] = lines[2][:-3] + ("0" if lines[2][-3] != "0" else "1") + lines[2][-2:]
        want = 3
    elif mode == "drop_line":
        del lines[1]
        want = 2
    else:
        lines[1], lines[2] = lines[2], lines[1]
        want = 2
    p.write_text("\n".join(lines) + "\n", encoding="ascii")
    rep = verify(p)
    assert not rep.ok and rep.first_bad == want, rep


def test_the_alert_is_on_disk_before_any_adapter_sees_it(tmp_path: Path) -> None:
    p = tmp_path / "spool.jsonl"
    seen = []

    class Probe:
        name = "probe"

        def deliver(self, rec):
            on_disk = [json.loads(r) for r in p.read_text(encoding="ascii").splitlines()]
            seen.append(on_disk[-1]["seq"] == rec["seq"] and on_disk[-1]["kind"] == rec["kind"])
            from engine.research.alerts.adapters import Delivery
            return Delivery("SENT", "probe")

    rec = emit(_alert(), [Probe()], path=p, clock=CLOCK)
    assert seen == [True]
    last = tail(1, p)[0]
    assert last["kind"] == "DELIVERY" and last["ref_seq"] == rec["seq"] and last["status"] == "SENT"


def test_an_adapter_that_raises_is_recorded_not_propagated(tmp_path: Path) -> None:
    class Boom:
        name = "boom"

        def deliver(self, rec):
            raise RuntimeError("down")

    p = tmp_path / "spool.jsonl"
    emit(_alert(), [Boom()], path=p, clock=CLOCK)
    assert tail(1, p)[0]["status"] == "FAILED" and "RuntimeError" in tail(1, p)[0]["detail"] and verify(p).ok


TG_ENV = {"ARQVIO_TG_TOKEN": "123:SECRET", "ARQVIO_TG_CHAT": "42"}


def test_the_telegram_adapter_contract() -> None:
    rec = {"seq": 3, "kind": "HEAT_HIGH", "severity": "warn", "payload": {"h": 0.07}, "source": "s"}
    assert TelegramAdapter(env={}).deliver(rec).status == "SKIPPED"
    assert "ARQVIO_TG_TOKEN" in TelegramAdapter(env={}).deliver(rec).detail
    calls = []
    ok = TelegramAdapter(sender=lambda url, data: calls.append((url, data)) or (200, '{"ok": true}'), env=TG_ENV)
    d = ok.deliver(rec)
    assert d.status == "SENT" and "SECRET" not in d.detail and calls[0][0].endswith("/bot123:SECRET/sendMessage")
    assert calls[0][1]["chat_id"] == "42" and "HEAT_HIGH seq 3" in calls[0][1]["text"]
    assert TelegramAdapter(sender=lambda u, d: (401, '{"ok":false}'), env=TG_ENV).deliver(rec).status == "FAILED"

    def boom(u, d):
        raise OSError("no route 123:SECRET")

    d = TelegramAdapter(sender=boom, env=TG_ENV).deliver(rec)
    assert d.status == "FAILED" and "SECRET" not in d.detail


SMTP_ENV = {"ARQVIO_SMTP_HOST": "smtp.x", "ARQVIO_SMTP_PORT": "587", "ARQVIO_SMTP_USER": "a@x",
            "ARQVIO_SMTP_PASSWORD": "pw-SECRET", "ARQVIO_SMTP_TO": "b@x"}


def test_the_email_adapter_contract() -> None:
    rec = {"seq": 1, "kind": "DRIFT_FIRE", "severity": "crit", "payload": {}, "source": "s"}
    d = EmailAdapter(env={"ARQVIO_SMTP_HOST": "h"}).deliver(rec)
    assert d.status == "SKIPPED" and "ARQVIO_SMTP_PASSWORD" in d.detail
    log = []

    class FakeSMTP:
        def __init__(self, host, port):
            log.append(("connect", host, port))

        def starttls(self):
            log.append("starttls")

        def login(self, user, pw):
            log.append(("login", user, pw))

        def send_message(self, msg):
            log.append(("send", msg["To"], msg["Subject"]))

        def quit(self):
            log.append("quit")

    d = EmailAdapter(smtp_factory=FakeSMTP, env=SMTP_ENV).deliver(rec)
    assert d.status == "SENT" and log[0] == ("connect", "smtp.x", 587) and log[-1] == "quit"
    assert ("send", "b@x", "[ARQVIO crit] DRIFT_FIRE seq 1") in log

    def refuse(host, port):
        raise ConnectionRefusedError("pw-SECRET")

    d = EmailAdapter(smtp_factory=refuse, env=SMTP_ENV).deliver(rec)
    assert d.status == "FAILED" and "SECRET" not in d.detail


def test_the_desktop_adapter_contract() -> None:
    rec = {"seq": 0, "kind": "HEAT_HIGH", "severity": "warn", "payload": {}, "source": "s"}
    assert DesktopAdapter(env={}).deliver(rec).status == "SKIPPED"
    argv = []
    d = DesktopAdapter(runner=lambda a: argv.append(a) or 0, env={"ARQVIO_DESKTOP_ALERTS": "1"}).deliver(rec)
    assert d.status == "SENT" and argv[0][:3] == ["msg.exe", "*", "/TIME:60"]
    assert DesktopAdapter(runner=lambda a: 1, env={"ARQVIO_DESKTOP_ALERTS": "1"}).deliver(rec).status == "FAILED"


def test_the_schema_refuses_unknowns() -> None:
    with pytest.raises(AlertSchemaError, match="unknown alert kind"):
        make_alert("NOPE", "warn", {}, "s")
    with pytest.raises(AlertSchemaError, match="severity"):
        make_alert("HEAT_HIGH", "loud", {}, "s")
    with pytest.raises(AlertSchemaError, match="source"):
        make_alert("HEAT_HIGH", "warn", {}, "")


def test_the_spool_path_resolves_at_call_time_and_creates_nothing(isolated_arqvio_home: Path) -> None:
    assert spool_path() == isolated_arqvio_home / "artifacts" / "alerts" / "spool.jsonl"
    assert not (isolated_arqvio_home / "artifacts" / "alerts").exists()


def test_a_drift_fire_becomes_an_audited_alert(tmp_path: Path) -> None:
    from engine.research.portfolio.drift import degraded_fixture, monitor

    p = tmp_path / "spool.jsonl"
    alerts = from_drift(monitor(degraded_fixture()), "artifacts/portfolio/drift.json", clock=CLOCK)
    for a in alerts:
        emit(a, [TelegramAdapter(env={})], path=p, clock=CLOCK)
    rows = tail(10, p)
    fire = [r for r in rows if r["kind"] == "DRIFT_FIRE"]
    assert len(fire) == 1 and fire[0]["payload"]["member"] == "m2" and fire[0]["payload"]["first_t"] == 700
    assert rows[-1]["kind"] == "DELIVERY" and rows[-1]["status"] == "SKIPPED" and verify(p).ok


def test_replay_heat_and_refusals_become_alerts() -> None:
    events = [{"t": 1, "kind": "HEAT_HIGH", "heat": 0.07, "unbounded": False, "per_symbol": {"A": 0.07}},
              {"t": 2, "kind": "CONFLICT", "symbol": "A", "members": ["x", "y"]},
              {"t": 3, "kind": "REFUSED", "member": "x", "trade": "t1", "cap": "HEAT", "values": {"heat": 0.11}}]
    out = from_replay(events, "replay.json", clock=CLOCK)
    assert [a["kind"] for a in out] == ["HEAT_HIGH", "REFUSED"] and out[1]["payload"]["cap"] == "HEAT"
