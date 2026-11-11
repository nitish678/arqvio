"""The GA finalists as read-only S6 parity SUBJECTS (the first consumer of the Stage-2 GA product).

``load_finalists`` reads ``<ARQVIO_HOME>/artifacts/ga/<campaign_id>/manifest_v2.json``
(resolved at call time), opens it read-only, and refuses anything it cannot
verify: another schema or version, another campaign id, a finalist count that
disagrees with the run block, a missing DNA or hash, a DNA outside the gene
space, or a recorded ``dna_hash`` that the DNA does not reproduce.

The finalists are engine-equivalence subjects and nothing more. Their status
is carried as CONTEXT text quoted verbatim from the S4 Stage-2 report; S6
re-judges nothing about edge.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from engine.core.paths import get_artifacts_dir
from engine.core.schema import same_schema
from engine.research.search.dna import DNA, validate_gene_space

GA_CAMPAIGN = "ga-usdcadm_h1-s7-cbcb2ae061e70bef"
MANIFEST_SCHEMA = "arqvio.ga.manifest"
MANIFEST_VERSION = 2
WEAK_CONTEXT = (
    'S4 Stage-2 report, verbatim: "The run is a **WEAK / no-DSR-survivor** outcome. It is not a discovery, '
    'and no structure claim is made." Context only: S6 proves engine equivalence and makes no edge claim.'
)


class SubjectError(ValueError):
    """A GA product that cannot be trusted as a parity subject source."""


@dataclass(frozen=True)
class Finalist:
    index: int
    dna: DNA
    dna_hash: str
    campaign_id: str
    run_hash: str
    datasets: dict[str, str]


def load_finalists(campaign_id: str = GA_CAMPAIGN) -> tuple[Finalist, ...]:
    path = get_artifacts_dir() / "ga" / campaign_id / "manifest_v2.json"
    if not path.is_file():
        raise SubjectError(f"no GA manifest at {path}")
    with path.open("r", encoding="utf-8") as handle:
        d = json.load(handle)
    if not same_schema(d.get("schema"), MANIFEST_SCHEMA) or d.get("version") != MANIFEST_VERSION:
        raise SubjectError(f"{path}: schema {d.get('schema')!r} v{d.get('version')!r}, "
                           f"expected {MANIFEST_SCHEMA} v{MANIFEST_VERSION}")
    run = d.get("run") or {}
    if run.get("campaign_id") != campaign_id:
        raise SubjectError(f"{path}: run.campaign_id {run.get('campaign_id')!r} is not {campaign_id!r}")
    cands = d.get("candidates") or []
    if len(cands) != run.get("finalists"):
        raise SubjectError(f"{path}: {len(cands)} candidates but run.finalists = {run.get('finalists')!r}")
    out = []
    for i, c in enumerate(cands):
        if "dna" not in c or "dna_hash" not in c:
            raise SubjectError(f"{path}: candidate {i} has no dna / dna_hash")
        try:
            validate_gene_space(c["dna"])
            dna = DNA(**c["dna"])
        except (TypeError, ValueError) as exc:
            raise SubjectError(f"{path}: candidate {i} DNA refused: {exc}") from exc
        if dna.dna_hash != c["dna_hash"]:
            raise SubjectError(f"{path}: candidate {i} records dna_hash {c['dna_hash']} but its DNA hashes to "
                               f"{dna.dna_hash}")
        out.append(Finalist(i, dna, dna.dna_hash, campaign_id, str(run.get("run_hash")), dict(c.get("dataset_id") or {})))
    if len({f.dna_hash for f in out}) != len(out):
        raise SubjectError(f"{path}: the finalists repeat a dna_hash")
    return tuple(out)


__all__ = ["Finalist", "GA_CAMPAIGN", "SubjectError", "WEAK_CONTEXT", "load_finalists"]
