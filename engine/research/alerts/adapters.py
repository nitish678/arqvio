"""Delivery adapters (DR §6): Telegram, desktop, email -- each with an INJECTED transport, each env-gated at CALL time.

``deliver(record)`` returns a ``Delivery``:
- SENT: the transport accepted the message;
- FAILED: a transport error or a refusal, with the detail;
- SKIPPED: the adapter's configuration is absent, naming the missing variables.

It never raises. Secrets are read from the environment when ``deliver`` runs (never at import) and never appear in a
detail string.
"""
from __future__ import annotations

import json
import os
import smtplib
import subprocess
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from email.message import EmailMessage

TG_VARS = ("ARQVIO_TG_TOKEN", "ARQVIO_TG_CHAT")
SMTP_VARS = ("ARQVIO_SMTP_HOST", "ARQVIO_SMTP_PORT", "ARQVIO_SMTP_USER", "ARQVIO_SMTP_PASSWORD", "ARQVIO_SMTP_TO")
DESKTOP_VAR = "ARQVIO_DESKTOP_ALERTS"


@dataclass(frozen=True)
class Delivery:
    status: str  # SENT | FAILED | SKIPPED
    detail: str


def message_text(rec: Mapping) -> str:
    payload = json.dumps(rec.get("payload", {}), sort_keys=True, separators=(",", ":"))[:600]
    return f"[ARQVIO {rec.get('severity')}] {rec.get('kind')} seq {rec.get('seq')}: {payload} (source {rec.get('source')})"


def _missing(env: Mapping[str, str], names: tuple[str, ...]) -> list[str]:
    return [n for n in names if not env.get(n)]


def _http_post_json(url: str, data: dict, timeout: float = 15.0) -> tuple[int, str]:
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 -- fixed https host below
        return resp.status, resp.read().decode("utf-8", errors="replace")


class TelegramAdapter:
    name = "telegram"

    def __init__(self, sender: Callable[[str, dict], tuple[int, str]] | None = None,
                 env: Mapping[str, str] | None = None) -> None:
        self.sender = sender or _http_post_json
        self.env = env

    def deliver(self, rec: Mapping) -> Delivery:
        env = os.environ if self.env is None else self.env
        missing = _missing(env, TG_VARS)
        if missing:
            return Delivery("SKIPPED", f"not configured: {', '.join(missing)} unset")
        url = f"https://api.telegram.org/bot{env['ARQVIO_TG_TOKEN']}/sendMessage"
        try:
            code, body = self.sender(url, {"chat_id": env["ARQVIO_TG_CHAT"], "text": message_text(rec)})
        except Exception as exc:  # noqa: BLE001 -- delivery never raises into the producer
            return Delivery("FAILED", f"transport error {type(exc).__name__}")
        ok = code == 200 and '"ok":true' in body.replace(" ", "")
        return Delivery("SENT" if ok else "FAILED", f"http {code}")


def _msg_runner(argv: list[str]) -> int:
    return subprocess.run(argv, check=False, timeout=30).returncode


class DesktopAdapter:
    name = "desktop"

    def __init__(self, runner: Callable[[list[str]], int] | None = None, env: Mapping[str, str] | None = None) -> None:
        self.runner = runner or _msg_runner
        self.env = env

    def deliver(self, rec: Mapping) -> Delivery:
        env = os.environ if self.env is None else self.env
        if env.get(DESKTOP_VAR) != "1":
            return Delivery("SKIPPED", f"not configured: {DESKTOP_VAR} != 1")
        try:
            rc = self.runner(["msg.exe", "*", "/TIME:60", message_text(rec)[:255]])
        except Exception as exc:  # noqa: BLE001
            return Delivery("FAILED", f"runner error {type(exc).__name__}")
        return Delivery("SENT" if rc == 0 else "FAILED", f"exit {rc}")


class EmailAdapter:
    name = "email"

    def __init__(self, smtp_factory: Callable[[str, int], object] | None = None,
                 env: Mapping[str, str] | None = None) -> None:
        self.smtp_factory = smtp_factory or (lambda host, port: smtplib.SMTP(host, port, timeout=20))
        self.env = env

    def deliver(self, rec: Mapping) -> Delivery:
        env = os.environ if self.env is None else self.env
        missing = _missing(env, SMTP_VARS)
        if missing:
            return Delivery("SKIPPED", f"not configured: {', '.join(missing)} unset")
        msg = EmailMessage()
        msg["Subject"] = f"[ARQVIO {rec.get('severity')}] {rec.get('kind')} seq {rec.get('seq')}"
        msg["From"] = env["ARQVIO_SMTP_USER"]
        msg["To"] = env["ARQVIO_SMTP_TO"]
        msg.set_content(message_text(rec))
        try:
            client = self.smtp_factory(env["ARQVIO_SMTP_HOST"], int(env["ARQVIO_SMTP_PORT"]))
            try:
                client.starttls()
                client.login(env["ARQVIO_SMTP_USER"], env["ARQVIO_SMTP_PASSWORD"])
                client.send_message(msg)
            finally:
                client.quit()
        except Exception as exc:  # noqa: BLE001
            return Delivery("FAILED", f"smtp error {type(exc).__name__}")
        return Delivery("SENT", "smtp accepted")


__all__ = ["DESKTOP_VAR", "Delivery", "DesktopAdapter", "EmailAdapter", "SMTP_VARS", "TG_VARS", "TelegramAdapter",
           "message_text"]
