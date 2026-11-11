"""Read-only implementation declarations, not executed parity or certification proof.

Kept outside registry.public(): adding evidence metadata must not change frozen
registry/DNA hashes. Runtime allowlists are authoritative for switch support.
"""
from __future__ import annotations

from typing import Any

from engine.research.genes import registry

TARGETS = ("oracle", "kernel", "live_ea", "tester_ea")


def matrix() -> dict[str, Any]:
    from engine.research.backtest.fast_path import SUPPORTED_SIGNALS
    from engine.research.backtest.signal_switches import READY_SIGNALS
    from engine.research.backtest.strategies.switch import READY_MGMT
    from engine.research.backtest.switch_gates import READY_BIAS, READY_FILTERS
    from engine.research.deploy.live_ea import LIVE_MGMT
    from engine.research.search.gene_pool import FAMILY_GENES
    from engine.research.backtest.signal_switches import combine
    import inspect

    declared_grammar = {
        "decision_time": "bar_close", "fill_time": "next_bar_open",
        "bias_combination": "all_active_biases_allow_direction",
        "filter_combination": "all_active_filters_allow_entry",
        "signal_combination": inspect.getdoc(combine),
        "supported_execution": ["MARKET2"],
        "configurable_quorum": False,
    }

    switches = {"bias_switches": READY_BIAS, "filter_switches": READY_FILTERS,
                "signal_switches": READY_SIGNALS, "mgmt_switches": READY_MGMT}
    rows = []
    for gene in registry.GENES:
        ready = gene.status is registry.Status.READY
        implemented = ready and (
            bool(gene.family and gene.family in SUPPORTED_SIGNALS)
            or gene.dna_field in ("sl_atr", "tp_atr", "risk_pct", "session", "trend_ema", "exec_mode")
            or gene.name.split(".", 1)[1] in switches.get(gene.dna_field, ()))
        fixed = gene.status is registry.Status.FIXED
        live = implemented or fixed or gene.status is registry.Status.EA_ONLY
        if gene.dna_field == "mgmt_switches":
            live = implemented and gene.name.split(".", 1)[1] in LIVE_MGMT
        # codegen._validated refuses gated and switched DNA; these are not tester capabilities.
        tester = implemented and (bool(gene.family) or gene.dna_field in
                                  ("sl_atr", "tp_atr", "risk_pct", "exec_mode"))
        declared = {"oracle": implemented or fixed, "kernel": implemented or fixed,
                    "live_ea": live, "tester_ea": tester or fixed}
        parameterization = ("family_parameters" if gene.family else "fixed_switch_rule" if gene.dna_field in switches
                            else "dna_field" if implemented else "cell_policy" if fixed else "not_available")
        rows.append({"name": gene.name, "legacy_status": gene.status.value,
                     "breedable_declared": bool(implemented),
                     "parameterization": parameterization, "declared_parameters": list(gene.params),
                     "breeding_fields": ([spec.name for spec in FAMILY_GENES[gene.family]] if gene.family
                                         else [gene.dna_field] if implemented else []),
                     "support": {target: {"declared": bool(declared[target]),
                                          "evidence_status": "UNVERIFIED", "proof_runs": []}
                                 for target in TARGETS},
                     "note": gene.note})
    return {"schema": "arqvio.gene_capabilities", "version": 1,
            "registry_hash": registry.registry_hash(), "declared_grammar": declared_grammar,
            "scope": "Implementation declarations only; no proof run loaded or certification implied",
            "genes": rows}


def require_breedable(dna) -> None:
    """Refuse unsupported active grammar before GA registry admission; no RNG/hash mutation."""
    from engine.research.backtest.signal_switches import READY_SIGNALS
    from engine.research.backtest.strategies.switch import READY_MGMT
    from engine.research.backtest.switch_gates import READY_BIAS, READY_FILTERS

    if dna.mgmt != "NONE":
        raise ValueError(f"legacy management {dna.mgmt!r} is not implemented; use supported mgmt_switches")
    groups = (("bias", dna.bias_switches, READY_BIAS),
              ("filter", dna.filter_switches, READY_FILTERS),
              ("signal", dna.signal_switches, READY_SIGNALS),
              ("management", dna.mgmt_switches, READY_MGMT),
              ("risk", dna.risk_switches, ()))
    for group, names, allowed in groups:
        for name in names:
            gene = registry.BY_NAME[f"{group}.{name}"]
            if name not in allowed or gene.status is not registry.Status.READY:
                raise ValueError(f"{gene.name} is not supported for breeding ({gene.status.value})")
