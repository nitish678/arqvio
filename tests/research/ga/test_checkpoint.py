"""Checkpoint / resume and the campaign monitor: exact, refusing, non-perturbing."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from engine.research.ga import checkpoint as ckpt
from engine.research.ga import runner as ga
from engine.research.ga.config import CAMPAIGN_GATES, IslandConfig, Selection, ga_config
from engine.research.ga.determinism import canonical, individuals_table
from engine.research.search import gene_pool as gp
from tests.research.ga.test_runner_integration import (  # noqa: F401
    SyntheticCell,
    _seed_isolated_home,
    real_artifacts,
)

HASH = "0123456789abcdef"


def _cfg(**kw):
    base = dict(seed=7, gates=CAMPAIGN_GATES, gene_pool=gp.default_gene_pool(), population=24,
                generations=5, trials_budget=500, final_n=6, selection=Selection(elitism=4))
    base.update(kw)
    return ga_config(**base)


def _outputs(res) -> str:
    return canonical({
        "table": individuals_table(res.archive), "generations": res.generations,
        "finalists": [i.effective_hash for i in res.finalists],
        "final_dsr": {k: v.__dict__ for k, v in res.final_dsr.items()},
        "stop": res.stop_reason, "trials": res.trials_used, "collisions": res.collisions,
    })


def _checkpointed(cfg, directory: Path):
    """Run with a checkpoint after every generation, one file per generation."""
    def save(loop):
        ckpt.save(directory / f"g{loop['generation']}.npz", loop=loop, extra={"note": "test"}, config_hash=HASH)
    return ga.search(cfg, SyntheticCell(), checkpoint=save)


def test_checkpointing_does_not_perturb_the_run(tmp_path: Path) -> None:
    assert _outputs(_checkpointed(_cfg(), tmp_path)) == _outputs(ga.search(_cfg(), SyntheticCell()))


@pytest.mark.parametrize("after", [0, 2, 3])
def test_a_resumed_run_is_the_uninterrupted_run(tmp_path: Path, after: int) -> None:
    full = _checkpointed(_cfg(), tmp_path)
    state = ckpt.load(tmp_path / f"g{after}.npz", config_hash=HASH)
    cell = SyntheticCell()
    resumed = ga.search(_cfg(), cell, resume=state)
    assert _outputs(resumed) == _outputs(full)
    assert cell.calls == full.trials_used - state["trials_used"]  # only the generations after the checkpoint ran


def test_a_stopped_run_resumes_into_nothing(tmp_path: Path) -> None:
    cfg = _cfg(trials_budget=24 + 20)
    full = _checkpointed(cfg, tmp_path)
    assert full.stop_reason == ga.STOP_BUDGET
    resumed = ga.search(cfg, SyntheticCell(), resume=ckpt.load(tmp_path / "g1.npz", config_hash=HASH))
    assert _outputs(resumed) == _outputs(full)


def test_finalist_returns_survive_the_checkpoint(tmp_path: Path) -> None:
    _checkpointed(_cfg(), tmp_path)
    state = ckpt.load(tmp_path / "g4.npz", config_hash=HASH)
    kept = [i for i in state["archive"] if i.result.per_period_returns is not None]
    assert {i.effective_hash for i in kept} == state["keep"] and kept
    assert all(isinstance(i.result.per_period_returns, np.ndarray) for i in kept)


@pytest.mark.parametrize("damage", ["flip", "truncate", "foreign", "empty"])
def test_a_damaged_checkpoint_is_refused_by_name(tmp_path: Path, damage: str) -> None:
    _checkpointed(_cfg(generations=2), tmp_path)
    path = tmp_path / "g1.npz"
    raw = path.read_bytes()
    if damage == "flip":  # inside the zip payload
        mid = len(raw) // 2
        path.write_bytes(raw[:mid] + bytes([raw[mid] ^ 0xFF]) + raw[mid + 1:])
    elif damage == "truncate":
        path.write_bytes(raw[: len(raw) // 3])
    elif damage == "foreign":
        np.savez(path, other=np.zeros(3))
    else:
        path.write_bytes(b"")
    with pytest.raises(ckpt.CheckpointError):
        ckpt.load(path, config_hash=HASH)


def test_an_edited_state_under_its_old_digest_is_refused(tmp_path: Path) -> None:
    """A structurally valid file whose state was changed: only the digest can tell."""
    _checkpointed(_cfg(generations=2), tmp_path)
    path = tmp_path / "g1.npz"
    with np.load(path) as z:
        members = {name: np.array(z[name]) for name in z.files}
    state = json.loads(members["state"].tobytes().decode())
    state["trials_used"] += 1
    members["state"] = np.frombuffer(json.dumps(state, sort_keys=True, separators=(",", ":")).encode(), dtype=np.uint8)
    np.savez(path, **members)
    with pytest.raises(ckpt.CheckpointError, match="digest mismatch"):
        ckpt.load(path, config_hash=HASH)


def test_a_checkpoint_of_another_config_is_refused(tmp_path: Path) -> None:
    _checkpointed(_cfg(generations=2), tmp_path)
    with pytest.raises(ckpt.CheckpointError, match="a resume must run the same configuration"):
        ckpt.load(tmp_path / "g1.npz", config_hash="fedcba9876543210")


def test_a_failed_write_leaves_the_previous_checkpoint_intact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _checkpointed(_cfg(generations=2), tmp_path)
    target = tmp_path / "g1.npz"
    before = target.read_bytes()
    state = ckpt.load(target, config_hash=HASH)
    loop = {"generation": 9, "rng": state["rng"], "registry": ga.Registry(seen=state["seen"]),
            "archive": state["archive"], "population": state["population"], "keep": state["keep"],
            "generations": state["generations"], "trials_used": state["trials_used"], "stop": state["stop"],
            "aborted": state["aborted"]}

    def crash(*a, **k):
        raise OSError("disk gone mid-rename")

    monkeypatch.setattr(ckpt.os, "replace", crash)
    with pytest.raises(OSError):
        ckpt.save(target, loop=loop, extra={}, config_hash=HASH)
    assert target.read_bytes() == before  # the old checkpoint, whole
    monkeypatch.undo()
    assert ckpt.load(target, config_hash=HASH)["generation"] == 1
    os.remove(target.with_name(target.name + ".tmp"))


def test_an_island_run_cannot_checkpoint_yet() -> None:
    cfg = _cfg(population=40, islands=IslandConfig(k=4))
    with pytest.raises(ga.GARunError, match="core loop only"):
        ga.run_pinned(cfg, "usdcadm_h1", checkpoint_path=Path("x.npz"), echo=lambda s: None)


# --- the real run: monitor + checkpoint must not perturb; resume reproduces ----

def _payload(out: Path) -> dict[str, str]:
    return {name: canonical(json.loads((out / name).read_text(encoding="utf-8"))) for name in ga.OUTPUT_FILES}


def test_monitor_and_checkpoint_do_not_perturb_a_real_run(real_artifacts: Path, isolated_arqvio_home: Path) -> None:  # noqa: F811
    _seed_isolated_home(real_artifacts, isolated_arqvio_home)
    cfg = _cfg(population=8, generations=2, trials_budget=40, final_n=3, selection=Selection(elitism=2))
    plain = ga.run_pinned(cfg, "usdcadm_h1", echo=lambda s: None)
    out = Path(plain["paths"]["manifest"]).parent
    reference = _payload(out)
    keep = isolated_arqvio_home / "ref"
    shutil.copytree(out, keep)
    point = isolated_arqvio_home / "ckpt.npz"
    ops = ga.run_pinned(cfg, "usdcadm_h1", echo=lambda s: None, checkpoint_path=point, monitor_every=0.0)
    assert ops["run_hash"] == plain["run_hash"] and _payload(out) == reference
    snap = json.loads((out / ga.MONITOR_FILE).read_text(encoding="utf-8"))
    assert snap["generation"] == 1 and snap["trials_used"] == 14 and {"funnel", "eta_seconds"} <= set(snap)
    manifest = json.loads((out / "manifest_v2.json").read_text(encoding="utf-8"))
    assert manifest["run"]["operational"]["checkpoint"] == str(point)
    resumed = ga.run_pinned(cfg, "usdcadm_h1", echo=lambda s: None, resume_path=point)
    assert resumed["run_hash"] == plain["run_hash"] and _payload(out) == reference
