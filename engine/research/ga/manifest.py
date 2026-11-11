"""Manifest v2: one complete, self-standing record per GA finalist.

S6 (verifier) and S8 (runtime) consume a candidate without re-running the
search, so each entry carries everything needed to say what it is, where it
came from and how far to trust it: the DNA and both hashes, its lineage,
its metrics and fitness on BOTH pinned cells, sham margins, DSR readings,
the campaign's degeneracy census, the controls it was judged against, the
audit and spread artifact each cell ran under, schema versions, the config
hash, the run hash, and each cell's dataset_id with its REQUIRED
dataset_provenance (campaign.manifest.dataset_provenance_for).

missing_candidate_fields() is the completeness check, run before anything is
written. A per-cell value may be None only where that cell's gates killed
the candidate -- and then its fitness record names the gate.

Completeness is per FINALIST (Q1, owner-ratified 2026-09-30). The finalists
carry complete per-cell records here. EVERY individual the run evaluated lives
in individuals.json (DNA, lineage, metrics, fitness: the run-hash table).
Dataset and provenance are run-level blocks -- run.cells.<cell> and each
cell's campaign Manifest -- deliberately not duplicated on every row.

Both DSR readings are kept (DSR recency policy): "primary_at_eval" is the
value that drove breeding, "primary_final" the reading at the run's own luck
bar, and reporting reads primary_final (fitness.py).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from engine.research.campaign.manifest import DATASET_PROVENANCE_VALUES
from engine.research.ga.determinism import plain_json

MANIFEST_SCHEMA = "arqvio.ga.manifest"
MANIFEST_VERSION = 2

REQUIRED_CANDIDATE_FIELDS: tuple[str, ...] = (
    "dna", "dna_hash", "effective_gene_hash", "lineage", "metrics", "sham_margin", "dsr",
    "cross_cell_delta", "degeneracy_census", "controls_id", "audit", "spread_sha256",
    "schema_versions", "config_hash", "run_hash", "dataset_id", "dataset_provenance",
)
# Keyed by cell: every entry must name both cells.
PER_CELL_FIELDS: tuple[str, ...] = (
    "metrics", "sham_margin", "audit", "spread_sha256", "dataset_id", "dataset_provenance",
)
REQUIRED_LINEAGE = ("generation_found", "parents", "operators")
REQUIRED_CONTROLS = ("kind", "salt", "regime", "engine", "seed")
REQUIRED_AUDIT = ("as_of", "time_basis", "source")


def _empty(value: Any) -> bool:
    return value is None or value == "" or value == {} or value == []


def missing_candidate_fields(entry: dict[str, Any], cells: tuple[str, str]) -> list[str]:
    """Every required field absent or empty, as dotted names; [] when complete."""
    missing = [name for name in REQUIRED_CANDIDATE_FIELDS if _empty(entry.get(name))]
    lineage = entry.get("lineage") or {}
    missing += [f"lineage.{k}" for k in REQUIRED_LINEAGE if k not in lineage]
    controls = entry.get("controls_id") or {}
    missing += [f"controls_id.{k}" for k in REQUIRED_CONTROLS if _empty(controls.get(k))]
    for name in PER_CELL_FIELDS:
        per_cell = entry.get(name) or {}
        for cell in cells:
            if cell not in per_cell:
                missing.append(f"{name}.{cell}")
    metrics = entry.get("metrics") or {}
    for cell in cells:
        m = metrics.get(cell) or {}
        fitness = m.get("fitness") or {}
        if "fitness" not in m or "killed_by" not in fitness:
            missing.append(f"metrics.{cell}.fitness")
            continue
        killed = fitness["killed_by"] is not None
        if not killed and (entry.get("sham_margin") or {}).get(cell) is None:
            missing.append(f"sham_margin.{cell}")
        for k in REQUIRED_AUDIT:
            if _empty(((entry.get("audit") or {}).get(cell) or {}).get(k)):
                missing.append(f"audit.{cell}.{k}")
        if _empty((entry.get("spread_sha256") or {}).get(cell)):
            missing.append(f"spread_sha256.{cell}")
        if _empty((entry.get("dataset_id") or {}).get(cell)):
            missing.append(f"dataset_id.{cell}")
        if (entry.get("dataset_provenance") or {}).get(cell) not in DATASET_PROVENANCE_VALUES:
            missing.append(f"dataset_provenance.{cell}")
    dsr = entry.get("dsr") or {}
    missing += [f"dsr.{k}" for k in ("primary_at_eval", "primary_final", "cross") if k not in dsr]
    return sorted(set(missing))


def build_manifest(
    *, run: dict[str, Any], candidates: list[dict[str, Any]], cells: tuple[str, str],
) -> dict[str, Any]:
    """The v2 document. Refuses (ValueError) if any candidate is incomplete."""
    problems = {
        c.get("dna_hash", f"#{i}"): missing
        for i, c in enumerate(candidates)
        if (missing := missing_candidate_fields(c, cells))
    }
    if problems:
        raise ValueError(f"manifest v2 incomplete: {problems}")
    return {
        "schema": MANIFEST_SCHEMA,
        "version": MANIFEST_VERSION,
        "cells": {"primary": cells[0], "cross": cells[1]},
        "run": run,
        "candidates": candidates,
    }


def write_json(path: Path, obj: Any) -> Path:
    """Sorted-key, indented JSON (non-finite floats as strings); parents created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Clock readings stay in the file: it records when it ran.
    text = json.dumps(plain_json(obj, drop_time=False), indent=2, sort_keys=True, allow_nan=False)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


__all__ = [
    "MANIFEST_SCHEMA",
    "MANIFEST_VERSION",
    "PER_CELL_FIELDS",
    "REQUIRED_CANDIDATE_FIELDS",
    "build_manifest",
    "missing_candidate_fields",
    "write_json",
]
