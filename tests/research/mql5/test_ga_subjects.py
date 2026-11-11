"""The GA finalists load read-only from manifest v2, verified, or are refused by name."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from engine.research.mql5 import ga_subjects as gs
from engine.research.mql5.magic import allocate

@pytest.fixture
def source_manifest(real_artifacts_dir: Path) -> Path:
    p = real_artifacts_dir / "ga" / gs.GA_CAMPAIGN / "manifest_v2.json"
    if not p.is_file():
        pytest.skip(f"the Stage-2 GA product is not on this machine ({p})")
    return p


@pytest.fixture
def real_manifest(source_manifest: Path, isolated_arqvio_home: Path) -> Path:
    """The real Stage-2 manifest, COPIED into the isolated home (the original is never opened for write)."""
    dest = isolated_arqvio_home / "artifacts" / "ga" / gs.GA_CAMPAIGN / "manifest_v2.json"
    dest.parent.mkdir(parents=True)
    shutil.copyfile(source_manifest, dest)
    return dest


def _write(home: Path, doc: dict) -> None:
    p = home / "artifacts" / "ga" / gs.GA_CAMPAIGN / "manifest_v2.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc), encoding="utf-8")


def test_the_twenty_finalists_load_verified_and_unchanged(real_manifest: Path, source_manifest: Path) -> None:
    before = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    fins = gs.load_finalists()
    assert len(fins) == 20 and [f.index for f in fins] == list(range(20))
    assert {f.dna.signal for f in fins} == {"MA_CROSS"}
    assert all(f.run_hash.startswith("4bc07ddbcde1e75c") for f in fins)
    assert fins[0].dna_hash == "4fb42548f31d2741"
    assert fins[0].datasets == {"usdcadm_h1": "aed3bf314539ec5a", "xauusdm_h1": "447b843f87438a9a"}
    assert hashlib.sha256(source_manifest.read_bytes()).hexdigest() == before  # read-only, by content
    magics = [allocate(gs.GA_CAMPAIGN, f.dna_hash, [g.dna_hash for g in fins]) for f in fins]
    assert magics == list(range(610_100_000, 610_100_020))


def test_the_context_is_quoted_not_judged() -> None:
    assert "WEAK / no-DSR-survivor" in gs.WEAK_CONTEXT and "no edge claim" in gs.WEAK_CONTEXT


@pytest.mark.parametrize("damage, needle", [
    ("schema", "schema"), ("foreign", "schema"), ("campaign", "run.campaign_id"),
    ("count", "candidates but run.finalists"), ("hash", "hashes to"), ("gene", "DNA refused"),
    ("missing", "has no dna"),
])
def test_an_untrustworthy_manifest_is_refused_by_name(real_manifest: Path, isolated_arqvio_home: Path,
                                                      damage: str, needle: str) -> None:
    doc = json.loads(real_manifest.read_text(encoding="utf-8"))
    bad = copy.deepcopy(doc)
    if damage == "schema":
        bad["version"] = 3
    elif damage == "foreign":
        bad["schema"] = "other.ga.manifest"
    elif damage == "campaign":
        bad["run"]["campaign_id"] = "ga-other"
    elif damage == "count":
        bad["candidates"] = bad["candidates"][:19]
    elif damage == "hash":
        bad["candidates"][3]["dna_hash"] = "0" * 16
    elif damage == "gene":
        bad["candidates"][3]["dna"]["donchian_period"] = 2
    else:
        del bad["candidates"][3]["dna"]
    _write(isolated_arqvio_home, bad)
    with pytest.raises(gs.SubjectError, match=needle):
        gs.load_finalists()


def test_a_missing_manifest_is_named(isolated_arqvio_home: Path) -> None:
    with pytest.raises(gs.SubjectError, match="no GA manifest"):
        gs.load_finalists()
