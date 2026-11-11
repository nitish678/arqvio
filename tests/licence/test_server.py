"""The licence server end to end through TestClient on a throwaway sqlite file.

WHY: the server's verdicts (400 / 403 / 409 / 429) are what the app will show users, so each is pinned here, and the
token it returns must verify with the engine's own token code.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from engine.licence.keys import make_key, parse
from engine.licence.limits import FREE_LIMITS, PAID_LIMITS
from engine.licence.token import hash_hwid, verify
from server.licence.app import create_app, key_secret, token_secret

MASTER = b"unit-test-master-secret"
PC_A, PC_B = "test-pc-a", "test-pc-b"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "licence.sqlite"


@pytest.fixture
def client(db_path: Path, clock: FakeClock) -> TestClient:
    return TestClient(create_app(db_path, MASTER, rate_limit=1000, clock=clock))


def _key(client: TestClient, plan: str = "free", **kwargs: int) -> str:
    return client.app.state.db.issue_key(plan, key_secret(MASTER), **kwargs)


def _activate(client: TestClient, key: str, hwid: str = PC_A):  # noqa: ANN202
    return client.post("/activate", json={"key": key, "hwid": hwid, "app_version": "3.9.0"})


def test_activate_ok_returns_a_verifiable_token(client: TestClient, clock: FakeClock) -> None:
    key = _key(client)
    response = _activate(client, key)
    assert response.status_code == 200
    body = response.json()
    payload = verify(body["token"], token_secret(MASTER), hwid_hash=hash_hwid(PC_A), now=int(clock.now))
    assert payload["key_serial"] == parse(key).serial
    assert payload["is_free"] is True and payload["seats"] == 1 and payload["licence_ended"] is False
    assert payload["limits"] == FREE_LIMITS.to_dict()


def test_raw_hwid_is_never_stored(client: TestClient, db_path: Path) -> None:
    _activate(client, _key(client))
    conn = sqlite3.connect(db_path)
    dump = "\n".join(conn.iterdump())
    conn.close()
    assert PC_A not in dump and hash_hwid(PC_A) in dump


def test_same_pc_can_activate_again(client: TestClient) -> None:
    key = _key(client)
    assert _activate(client, key).status_code == 200
    assert _activate(client, key).status_code == 200


def test_second_pc_over_seat_limit_is_409_then_reset_frees(client: TestClient) -> None:
    key = _key(client)
    assert _activate(client, key, PC_A).status_code == 200
    refused = _activate(client, key, PC_B)
    assert refused.status_code == 409 and refused.json()["detail"] == "KEY IN USE ON ANOTHER PC"
    # The oldest machine was not evicted.
    assert client.post("/refresh", json={"key": key, "hwid": PC_A}).status_code == 200
    reset = client.post("/reset", json={"key": key})
    assert reset.status_code == 200 and reset.json()["freed"] == 1
    assert _activate(client, key, PC_B).status_code == 200
    assert client.post("/refresh", json={"key": key, "hwid": PC_A}).status_code == 404


def test_paid_plan_with_two_seats(client: TestClient, clock: FakeClock) -> None:
    key = _key(client, "pro", seats=2)
    assert _activate(client, key, PC_A).status_code == 200
    body = _activate(client, key, PC_B).json()
    payload = verify(body["token"], token_secret(MASTER), hwid_hash=hash_hwid(PC_B), now=int(clock.now))
    assert payload["is_free"] is False and payload["limits"] == PAID_LIMITS.to_dict()
    assert _activate(client, key, "test-pc-c").status_code == 409


def test_ended_plan_is_403(client: TestClient) -> None:
    key = _key(client)
    client.app.state.db.set_plan_ended(parse(key).serial)
    response = _activate(client, key)
    assert response.status_code == 403 and response.json()["detail"] == "PLAN ENDED"


@pytest.mark.parametrize("bad", ["", "nonsense", "ABCD-1234567890-0123456789ABCDEF"])
def test_invalid_key_is_400(client: TestClient, bad: str) -> None:
    response = _activate(client, bad)
    assert response.status_code == 400 and response.json()["detail"] == "INVALID KEY"


def test_genuine_but_unissued_key_is_400(client: TestClient) -> None:
    unissued = str(make_key(1, key_secret(MASTER)))
    assert _activate(client, unissued).status_code == 400


def test_key_from_another_server_is_400(client: TestClient) -> None:
    real = parse(_key(client))
    forged = str(make_key(real.serial, key_secret(b"some-other-master")))
    assert _activate(client, forged).status_code == 400


def test_aggressive_key_input_accepted(client: TestClient) -> None:
    key = _key(client)
    assert _activate(client, "  " + key.lower().replace("-", " ") + "\n").status_code == 200


def test_refresh_reissues_and_carries_the_ended_verdict(client: TestClient, clock: FakeClock) -> None:
    key = _key(client)
    _activate(client, key)
    clock.now += 3600
    body = client.post("/refresh", json={"key": key, "hwid": PC_A}).json()
    payload = verify(body["token"], token_secret(MASTER), hwid_hash=hash_hwid(PC_A), now=int(clock.now))
    assert payload["iat"] == int(clock.now)
    client.app.state.db.set_plan_ended(parse(key).serial)
    ended = client.post("/refresh", json={"key": key, "hwid": PC_A})
    assert ended.status_code == 200 and ended.json()["licence_ended"] is True
    assert client.post("/refresh", json={"key": key, "hwid": PC_B}).status_code == 404


def test_plan_lookup(client: TestClient) -> None:
    key = _key(client)
    _activate(client, key)
    serial = parse(key).serial
    body = client.get(f"/plan/{serial}").json()
    assert body == {"serial": serial, "plan": "free", "is_free": True, "seats": 1, "seats_used": 1, "ended": False}
    assert client.get("/plan/0000000000").status_code == 404
    assert client.get("/plan/abc").status_code == 404


def test_updatecheck(client: TestClient) -> None:
    db = client.app.state.db
    assert client.get("/updatecheck", params={"version": "3.9.0"}).json() == {}
    db.add_release("3.9.0", "https://example.invalid/arqvio-3.9.0.exe")
    db.add_release("3.10.0", "https://example.invalid/arqvio-3.10.0.exe")
    newer = client.get("/updatecheck", params={"version": "3.9.0"}).json()
    assert newer == {"latest": "3.10.0", "url": "https://example.invalid/arqvio-3.10.0.exe"}
    assert client.get("/updatecheck", params={"version": "3.10.0"}).json() == {}
    assert client.get("/updatecheck", params={"version": "3.10"}).json() == {}


def test_rate_limit_429_and_window_reset(db_path: Path, clock: FakeClock) -> None:
    client = TestClient(create_app(db_path, MASTER, clock=clock))  # default: 30 a minute
    statuses = [client.get("/updatecheck").status_code for _ in range(31)]
    assert statuses[:30] == [200] * 30 and statuses[30] == 429
    assert client.get("/updatecheck").json() == {"detail": "TOO MANY REQUESTS"}
    clock.now += 61
    assert client.get("/updatecheck").status_code == 200


def test_schema_survives_reopen(db_path: Path, clock: FakeClock) -> None:
    first = TestClient(create_app(db_path, MASTER, clock=clock))
    key = _key(first)
    second = TestClient(create_app(db_path, MASTER, clock=clock))
    assert _activate(second, key).status_code == 200
