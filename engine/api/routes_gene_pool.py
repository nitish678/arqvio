"""Gene pool endpoints: read, replace and patch the pool the sampler draws from.

    GET   /api/v1/gene-pool/schema -> families, per-family gene ranges, constraints
    GET   /api/v1/gene-pool/registry -> every gene both engines know, its origin and status (genes/registry.py)
    GET   /api/v1/gene-pool        -> the stored pool (default when none stored)
    PUT   /api/v1/gene-pool        -> replace with a full pool
    PATCH /api/v1/gene-pool        -> merge a partial pool, validated as a whole

Bodies are untyped JSON checked by gene_pool.canonicalise, so every semantic
error comes back as 422 with the same {"error", "problems"} detail.
"""
from __future__ import annotations

import dataclasses
import threading
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from engine.research.genes import registry
from engine.research.search import gene_pool

router = APIRouter(prefix="/api/v1/gene-pool", tags=["gene-pool"])

# PATCH is read-merge-write; two concurrent writers must not interleave.
_WRITE_LOCK = threading.Lock()


def _invalid(exc: gene_pool.GenePoolError) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={"error": "invalid gene pool", "problems": exc.problems},
    )


def _stored() -> gene_pool.GenePool:
    try:
        return gene_pool.load()
    except gene_pool.GenePoolError as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "stored gene pool is invalid", "problems": exc.problems},
        ) from exc


@router.get("/schema")
def get_schema() -> dict[str, Any]:
    return gene_pool.schema()


@router.get("/registry")
def get_registry() -> dict[str, Any]:
    return registry.public()


@router.get("/capabilities")
def get_capabilities() -> dict[str, Any]:
    from engine.research.genes.capabilities import matrix

    return matrix()


@router.get("")
def get_pool() -> dict[str, Any]:
    return _stored().to_dict()


def _refuse_while_a_campaign_runs() -> None:
    """A saved pool changes the config hash of a campaign whose child is still starting (S5-APP H-START LAW)."""
    from engine.api.routes_app_campaigns import live, refuse

    if live():
        raise refuse(f"the gene pool cannot change while a campaign process runs ({', '.join(live())})", run=None,
                     cell=None, step="gene-pool")


@router.get("/smart-priors")
def smart_priors() -> dict[str, Any]:
    """Suggested family weights from the vault, half of them uniform (search/smart_priors.py). Read-only: apply them
    with PATCH."""
    from engine.research.search.smart_priors import suggest
    from engine.research.vault import library

    return suggest(_stored(), library.load())


@router.put("")
def put_pool(body: Any = Body(...)) -> dict[str, Any]:
    _refuse_while_a_campaign_runs()
    try:
        pool = gene_pool.canonicalise(body)
        if isinstance(body, dict) and "gates" not in body:
            # the gates have their own control (Gene pool screen, Entry gates): a PUT without them keeps them
            pool = dataclasses.replace(pool, gates=_stored().gates)
    except gene_pool.GenePoolError as exc:
        raise _invalid(exc) from exc
    with _WRITE_LOCK:
        gene_pool.save(pool)
    return pool.to_dict()


@router.patch("")
def patch_pool(body: Any = Body(...)) -> dict[str, Any]:
    _refuse_while_a_campaign_runs()
    with _WRITE_LOCK:
        current = _stored()
        try:
            pool = gene_pool.merge(current, body)
        except gene_pool.GenePoolError as exc:
            raise _invalid(exc) from exc
        gene_pool.save(pool)
    return pool.to_dict()


__all__ = ["router"]
