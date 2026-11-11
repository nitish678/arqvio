"""The read-only account probe and the status snapshot (DR §7).

Per account, in config order, one MetaTrader5 connection at a time:
1. ``initialize(path, login, server[, password from password_env when it is set])``;
2. ``account_info``: a DEMO account is probed; a REAL account is probed only if its owner armed it
   (``engine.research.accounts.arming``), otherwise it is UNAVAILABLE with the arm command as the reason;
3. balance / equity / margin / free margin, ``positions_total`` / ``orders_total`` (COUNTS only);
4. the basis label (``basis.probe_with_retry``) on the account's probe symbol;
5. spec drift: the live ``symbol_info`` of the probe symbol against the pinned spec it maps to;
6. staleness: now_utc - the last tick's UTC time (through the basis label) > STALE_S is STALE;
7. ``shutdown``.

The probe calls no order function; ``tests/research/health/test_health.py`` checks the package's source for it.
Every path -- the accounts config (``<ARQVIO_HOME>/config/accounts.json``) and the snapshot
(``<ARQVIO artifacts>/health/status.json``) -- resolves at call time.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.core.paths import get_artifacts_dir, get_arqvio_home
from engine.research.data.time_basis import UTC_SKEW_TOLERANCE_S
from engine.research.health.basis import BasisLabel, BasisRefused, probe_with_retry, to_utc
from engine.research.accounts.arming import is_armed

STALE_S = 300
DEMO, REAL = 0, 2
SPEC_FIELDS = (("digits", "digits"), ("point", "point"), ("trade_contract_size", "contract_size"),
               ("trade_tick_size", "trade_tick_size"), ("trade_tick_value_profit", "trade_tick_value_profit"),
               ("trade_tick_value_loss", "trade_tick_value_loss"), ("volume_min", "volume_min"),
               ("volume_step", "volume_step"), ("volume_max", "volume_max"))
SCHEMA = "arqvio.health.status"


class HealthError(RuntimeError):
    """A configuration the probe refuses."""


def accounts_path() -> Path:
    return get_arqvio_home() / "config" / "accounts.json"


def load_accounts(path: Path | None = None) -> list[dict]:
    path = path or accounts_path()
    if not path.is_file():
        raise HealthError(f"no accounts config at {path}")
    accts = json.loads(path.read_text(encoding="utf-8"))["accounts"]
    for a in accts:
        if "password" in a:
            raise HealthError(f"{a.get('id')}: a literal password in the config is refused; use password_env")
        for k in ("id", "login", "server", "terminal", "password_env", "probe_symbol", "pinned_symbol"):
            if k not in a:
                raise HealthError(f"{a.get('id')}: config field {k!r} missing")
    if len({int(a["login"]) for a in accts}) != len(accts):
        raise HealthError("two configured accounts share a login")
    return accts


def snapshot_path() -> Path:
    """Resolved at call time; creates nothing (``write_snapshot`` creates the directory it writes into)."""
    return get_artifacts_dir() / "health" / "status.json"


def _spec_drift(live: Any, pinned: Any) -> dict:
    out = {}
    for live_f, pin_f in SPEC_FIELDS:
        lv, pv = getattr(live, live_f, None), getattr(pinned, pin_f, None)
        if lv != pv:
            out[pin_f] = {"live": lv, "pinned": pv}
    return out


def probe_account(mt5: Any, acct: dict, *, pinned_spec: Callable[[str], Any],
                  env: dict | None = None, sleep: Callable[[float], None] | None = None,
                  now: Callable[[], datetime] = lambda: datetime.now(tz=UTC), arming: Path | None = None) -> dict:
    env = os.environ if env is None else env
    kw = {"path": acct["terminal"], "login": int(acct["login"]), "server": acct["server"]}
    if env.get(acct["password_env"]):
        kw["password"] = env[acct["password_env"]]
    row: dict[str, Any] = {"id": acct["id"], "login": int(acct["login"]), "server": acct["server"],
                           "password_source": "env" if "password" in kw else "terminal saved login"}
    if not mt5.initialize(**kw):
        return row | {"status": "UNAVAILABLE", "reason": f"initialize failed: {mt5.last_error()}",
                      "probed_utc": now().isoformat()}
    try:
        a = mt5.account_info()
        if a is None:
            return row | {"status": "UNAVAILABLE", "reason": f"account_info None: {mt5.last_error()}",
                          "probed_utc": now().isoformat()}
        if int(a.trade_mode) not in (DEMO, REAL) or int(a.login) != int(acct["login"]):
            return row | {"status": "UNAVAILABLE", "reason": f"trade_mode {a.trade_mode} login {a.login}",
                          "probed_utc": now().isoformat()}
        real = int(a.trade_mode) == REAL
        if real and not is_armed(int(a.login), str(a.server), arming):
            return row | {"status": "UNAVAILABLE", "real": True, "armed": False, "probed_utc": now().isoformat(),
                          "reason": f"real-money account not armed (login {a.login}, server {a.server}); the owner arms "
                                    f"it once: python -m engine.research.accounts.arming arm --login {a.login} "
                                    f"--server {a.server}"}
        row |= {"demo": not real, "real": real, "trade_mode": int(a.trade_mode), "margin_mode": int(a.margin_mode),
                "currency": a.currency, "balance": float(a.balance), "equity": float(a.equity),
                "margin": float(a.margin), "margin_free": float(a.margin_free),
                "positions": int(mt5.positions_total()), "orders": int(mt5.orders_total()),
                "probe_symbol": acct["probe_symbol"], "pinned_symbol": acct["pinned_symbol"]}
        mt5.symbol_select(acct["probe_symbol"], True)
        label: BasisLabel = probe_with_retry(mt5, acct["probe_symbol"], now=now,
                                             **({} if sleep is None else {"sleep": sleep}))
        row["basis"] = label.as_dict()
        info = mt5.symbol_info(acct["probe_symbol"])
        row["spec_drift"] = None if info is None else _spec_drift(info, pinned_spec(acct["pinned_symbol"]))
        tick = mt5.symbol_info_tick(acct["probe_symbol"])
        try:
            age = now().timestamp() - to_utc(int(tick.time), label)
            row |= {"last_tick_age_s": round(age, 3), "status": "STALE" if age > STALE_S else "OK"}
        except BasisRefused as exc:
            if all(d < -UTC_SKEW_TOLERANCE_S for d in label.deltas_s):
                # Every probe saw a tick older than now (S7 DEV-4): no fresh tick, so the clock is unmeasurable
                # and the market is stale. The label stays 'unresolved': hour-keyed rules still refuse.
                row |= {"last_tick_age_s": None, "status": "STALE",
                        "reason": f"clock unmeasurable: the last tick is {-label.deltas_s[-1]:.0f} s behind now ({exc})"}
            else:
                row |= {"last_tick_age_s": None, "status": "UNAVAILABLE", "reason": str(exc)}
        row["probed_utc"] = now().isoformat()
        return row
    finally:
        mt5.shutdown()


def snapshot(mt5: Any, *, accounts: list[dict] | None = None, pinned_spec: Callable[[str], Any] | None = None,
             env: dict | None = None, sleep: Callable[[float], None] | None = None,
             now: Callable[[], datetime] = lambda: datetime.now(tz=UTC), arming: Path | None = None) -> dict:
    if pinned_spec is None:
        from engine.research.search.real_backtest import spec_for
        pinned_spec = spec_for
    accts = load_accounts() if accounts is None else accounts
    rows = [probe_account(mt5, a, pinned_spec=pinned_spec, env=env, sleep=sleep, now=now, arming=arming)
            for a in accts]
    return {"schema": SCHEMA, "version": 1, "snapshot_utc": now().isoformat(), "accounts": rows}


def write_snapshot(snap: dict, path: Path | None = None) -> Path:
    path = path or snapshot_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(snap, indent=2, sort_keys=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return path


def main() -> int:
    import MetaTrader5 as mt5

    snap = snapshot(mt5)
    path = write_snapshot(snap)
    for r in snap["accounts"]:
        print(f"{r['id']} login {r['login']} {r['server']}: status {r['status']} demo {r.get('demo')} "
              f"basis {r.get('basis')} tick age {r.get('last_tick_age_s')} positions {r.get('positions')} "
              f"orders {r.get('orders')} spec drift {r.get('spec_drift')}")
    print(f"WROTE {path}")
    return 0 if all(r["status"] in ("OK", "STALE") for r in snap["accounts"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
