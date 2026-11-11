"""A reader takes a schema name or its pre-rename astra. spelling, nothing else."""
from __future__ import annotations

import pytest

from engine.core.schema import same_schema


@pytest.mark.parametrize(("found", "expected", "ok"), [
    ("arqvio.ga.manifest", "arqvio.ga.manifest", True),
    ("astra.ga.manifest", "arqvio.ga.manifest", True),       # written before the rename (a7127b9)
    ("astra.vault", "arqvio.ga.manifest", False),
    ("ASTRA.ga.manifest", "arqvio.ga.manifest", False),
    ("other.ga.manifest", "arqvio.ga.manifest", False),
    (None, "arqvio.vault", False),
    (2, "arqvio.vault", False),
    ("astra.thing", "plain-name", False),                    # only arqvio.* names have an old spelling
])
def test_same_schema(found, expected, ok) -> None:
    assert same_schema(found, expected) is ok
