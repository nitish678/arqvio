"""Control campaign scope: identical to the real campaign on every axis
except signal meaning.
"""
from __future__ import annotations

from collections.abc import Mapping


class ControlScopeMismatchError(RuntimeError):
    """Raised when the control campaign drifts on any scope field."""


_SCOPE_FIELDS: tuple[str, ...] = (
    "symbol",
    "timeframe",
    "dataset_hash",
    "cost_model",
    "risk_rules",
    "exec_mode",
    "min_trades",
    "n_trials",
    "date_from_utc",
    "date_to_utc",
)


def scope_to_control(real: Mapping[str, object]) -> dict[str, object]:
    """Construct a control scope that matches the real campaign 1:1.

    The control uses a different dna_hash (the search is over random
    nonsense signals) and a different ``search_id``; everything else
    must be identical.
    """
    out = dict(real)
    out["dna_hash_prefix"] = "ctrl-"  # only the DNA family differs
    return out


def assert_scopes_match(
    real: Mapping[str, object], control: Mapping[str, object]
) -> None:
    """Raise on any field drift. Risk rules and cost model must compare
    deep-equal, not just reference-equal."""
    for field in _SCOPE_FIELDS:
        if field not in real:
            raise ControlScopeMismatchError(f"real scope missing required field: {field!r}")
        if field not in control:
            raise ControlScopeMismatchError(
                f"control scope missing required field: {field!r}"
            )
        rv = real[field]
        cv = control[field]
        if isinstance(rv, Mapping) and isinstance(cv, Mapping):
            if dict(rv) != dict(cv):
                raise ControlScopeMismatchError(
                    f"scope drift on {field!r}: real={rv!r} control={cv!r}"
                )
            continue
        if rv != cv:
            raise ControlScopeMismatchError(
                f"scope drift on {field!r}: real={rv!r} control={cv!r}"
            )
    control_n = int(str(control["n_trials"]))
    real_n = int(str(real["n_trials"]))
    if control_n < real_n:
        raise ControlScopeMismatchError(
            f"control has fewer trials ({control['n_trials']}) "
            f"than real ({real['n_trials']})"
        )


def scope_from_campaign(
    *,
    symbol: str,
    timeframe: str,
    dataset_hash: str,
    cost_model: str,
    risk_rules: dict[str, object],
    exec_mode: str,
    min_trades: int,
    n_trials: int,
    date_from_utc: str,
    date_to_utc: str,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "dataset_hash": dataset_hash,
        "cost_model": cost_model,
        "risk_rules": risk_rules,
        "exec_mode": exec_mode,
        "min_trades": min_trades,
        "n_trials": n_trials,
        "date_from_utc": date_from_utc,
        "date_to_utc": date_to_utc,
    }


__all__ = [
    "ControlScopeMismatchError",
    "assert_scopes_match",
    "scope_from_campaign",
    "scope_to_control",
]
