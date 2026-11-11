"""The re-proof checks, the BLOCKED reason enum and the proof document (DR-2026-10-08-u3-worker §3-§6).

``reproof(cell_id, out_dir=..., battery_log=...)`` loads the registry FIRST (every SL4 tripwire), refuses an unknown
or non-VALIDATED cell, then selects its tier from the entry's campaign_status alone (DR-2026-10-09-u4-wizard §3), never
from a flag:
- RUN@<campaign>: the full tier, nine checks; the proof and status (PROVEN@<sha>) are U3's, unchanged;
- UNRUN: the pre-campaign tier, the seven PRECAMPAIGN_CHECKS, only when power_check is BLOCKED:no-campaign and no
  campaign directory or cited manifest exists for the cell (else it refuses: tier-contradiction). power and rederive
  are not run: the proof records them under not_applicable with their reasons, never as passes. Its status is
  PROVEN-PRECAMPAIGN@<sha>.
The checks run in a fixed order. Every check runs and records why it failed; nothing is defaulted or approximated.
The verdict is PROVEN when all pass, else BLOCKED:<reason> -- the first failing reason in REASON_ORDER. The proof is
canonical JSON with no clock reading, so a re-run on the same HEAD and inputs is byte-identical. The only writes are
<out>/<cell_id>.proof.json, one line appended to <out>/audit.jsonl, and the canary's temporary sqlite (a
TemporaryDirectory, removed).
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from engine.core import paths
from engine.research.power.status import power_status
from engine.research.registry import cells as registry
from engine.research.registry.annualisation import bars_per_year

PROOF_SCHEMA = "arqvio.reproof"
PROOF_VERSION = 1
# The BLOCKED reason enum (ARQVIO_U3_PROMPT_v1.md Part 1.3), in verdict precedence (DR §3). battery-fail and
# evidence-missing carry a suffix: battery-fail:<family|log|canary>, evidence-missing:<n>.
REASON_ORDER = ("dataset-missing", "hash-mismatch", "power-unproven", "battery-fail", "evidence-missing")
STATUS_GRAMMAR = re.compile(r"UNKNOWN|PROVEN@[0-9a-f]{7}|PROVEN-PRECAMPAIGN@[0-9a-f]{7}|BLOCKED:(dataset-missing|"
                            r"hash-mismatch|power-unproven|battery-fail:[A-Za-z_]+|evidence-missing:[1-9][0-9]*)")
# The pre-campaign tier (#2-R2, #3-Q3): what a never-searched cell can prove, and why the other two cannot run.
PRECAMPAIGN_CHECKS = ("dataset", "specs_spread", "audit_row", "annualisation", "evidence", "battery", "canary")
NOT_APPLICABLE_PRECAMPAIGN = {
    "power": "no campaign: U2's power check needs a campaign's hosts, prior and N (#2-R1); power_check is "
             "BLOCKED:no-campaign",
    "rederive": "no campaign: there are no recorded candidate metrics to re-derive (#2-R2)",
}
# The banked S6 parity battery (artifacts/reports/s6_report.md; artifacts/reports/u2/closing_parity_battery.log),
# family -> (fired, trades (slow), mismatches). DR §6.
BANKED_BATTERY: dict[str, tuple[str, int, int]] = {
    "RSI_REVERSION": ("47/60", 704, 0),
    "STOCH_CROSS": ("60/60", 1096, 0),
    "MACD_CROSS": ("60/60", 2967, 0),
    "DONCHIAN_BREAKOUT": ("60/60", 1994, 0),
}
BATTERY_TEST = "tests/research/backtest/test_family_kernel_parity.py"
BATTERY_MODULE = "tests.research.backtest.test_family_kernel_parity"
CANARY_BUDGET_S = 60.0  # scripts/canary_fast.py's budget
METRICS = ("sharpe", "n_trades", "total_return", "max_dd", "blocked_max_spread", "costs")
# The reason a check records when it raises instead of returning (a defect in its input, never a pass).
_ON_ERROR = {"dataset": "hash-mismatch", "specs_spread": "hash-mismatch", "audit_row": "hash-mismatch",
             "annualisation": "hash-mismatch", "power": "power-unproven", "evidence": "evidence-missing:1",
             "rederive": "hash-mismatch", "battery": "battery-fail:log", "canary": "battery-fail:canary"}
_MANIFEST_REF = re.compile(r"<ARQVIO_HOME>/artifacts/ga/[^/\s]+/manifest_v2\.json")
_LINE_REF = re.compile(r"^(?P<path>[^\s:]+):(?P<line>\d+)\b")


class ReproofRefused(Exception):
    """The re-proof refuses to run: an untrusted registry, or a cell it does not hold as VALIDATED."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def canonical(doc: Any) -> str:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def resolve_ref(ref: str) -> Path:
    """'<ARQVIO_HOME>/x' -> ARQVIO_HOME/x; anything else -> PROJECT_ROOT/x. Never a fallback between the two."""
    if ref.startswith("<ARQVIO_HOME>/"):
        return paths.get_arqvio_home() / ref[len("<ARQVIO_HOME>/"):]
    return paths.PROJECT_ROOT / ref


def git_head() -> str:
    return subprocess.run(["git", "-C", str(paths.PROJECT_ROOT), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def _ok(**detail: Any) -> dict[str, Any]:
    return {"ok": True, "reason": None, **detail}


def _fail(reason: str, **detail: Any) -> dict[str, Any]:
    return {"ok": False, "reason": reason, **detail}


def check_dataset(entry: dict[str, Any]) -> dict[str, Any]:
    from engine.research.data.dataset_snapshot import dataset_hash, datasets_dir

    sym, tf, h16 = entry["symbol"], entry["timeframe"], entry["dataset_hash16"]
    npz, side = datasets_dir() / f"{sym}_{tf}_{h16}.npz", datasets_dir() / f"{sym}_{tf}_{h16}.json"
    missing = [p.name for p in (npz, side) if not p.is_file()]
    if missing:
        return _fail("dataset-missing", missing=missing)
    try:  # a present but corrupt or tampered snapshot is a mismatch, never a crash and never "missing"
        with np.load(npz) as z:
            arrays = {k: z[k] for k in z.files}
        recomputed = dataset_hash(symbol=sym, timeframe=tf, time=arrays["time"], open_=arrays["open"],
                                  high=arrays["high"], low=arrays["low"], close=arrays["close"],
                                  spread=arrays["spread"])
        meta = json.loads(side.read_text(encoding="utf-8"))
        t = np.asarray(arrays["time"], dtype=np.int64)
        iso = lambda s: datetime.fromtimestamp(int(s), tz=UTC).isoformat()  # noqa: E731
        observed = {"start": iso(t[0]), "end": iso(t[-1]), "bars": int(t.size)}
    except Exception as exc:
        return _fail("hash-mismatch", npz_sha256=sha256_file(npz), sidecar_sha256=sha256_file(side),
                     error=f"{type(exc).__name__}: {exc}")
    sidecar = {"start": meta.get("start_utc"), "end": meta.get("end_utc"), "bars": meta.get("n_bars")}
    detail = {"npz_sha256": sha256_file(npz), "sidecar_sha256": sha256_file(side), "recomputed_hash16": recomputed,
              "npz_window": observed, "sidecar_window": sidecar, "sidecar_timeframe": meta.get("timeframe")}
    if recomputed != h16 or observed != entry["dataset_window"] or sidecar != entry["dataset_window"] \
            or meta.get("timeframe") != tf or meta.get("dataset_id") != h16:
        return _fail("hash-mismatch", **detail)
    return _ok(**detail)


def check_specs_spread(entry: dict[str, Any]) -> dict[str, Any]:
    from engine.config.settings import get_settings
    from engine.data.spread_calibration import spread_dir
    from engine.research.data.symbol_specs import specs_dir

    files = {"spread": spread_dir() / f"{entry['symbol']}.json", "specs": specs_dir() / f"{entry['symbol']}.json"}
    missing = [k for k, p in files.items() if not p.is_file()]
    if missing:
        return _fail("dataset-missing", missing=missing)
    got = {k: sha256_file(p) for k, p in files.items()}
    want = {"spread": entry["spread_artifact_hash"]["sha256"], "specs": entry["specs_artifact_hash"]["sha256"]}
    override = dict(get_settings().max_spread_points_by_symbol)
    detail = {"sha256": got, "spread_gate_override": override}
    if got != want or override:
        return _fail("hash-mismatch", **detail)
    return _ok(**detail)


def check_audit_row(entry: dict[str, Any]) -> dict[str, Any]:
    ref = entry["audit_row_ref"]
    p = resolve_ref(ref["file"])
    if not p.is_file():
        return _fail("evidence-missing:1", missing=[ref["file"]])
    lines = p.read_text(encoding="utf-8").splitlines()
    at = lines[ref["line"] - 1] if 0 < ref["line"] <= len(lines) else None
    rows = [n for n, line in enumerate(lines, 1) if line.startswith(f"{entry['symbol']},{entry['timeframe']},")]
    as_of = f"# AS_OF {ref['as_of']}" in lines
    detail = {"file_sha256": sha256_file(p), "row_at_line": at == ref["row"], "cell_rows": rows, "as_of": as_of,
              "mode": "recorded, unchanged (the audit needs live MT5 to recompute)"}
    if at != ref["row"] or rows != [ref["line"]] or not as_of:
        return _fail("hash-mismatch", **detail)
    return _ok(**detail)


def check_annualisation(entry: dict[str, Any]) -> dict[str, Any]:
    from engine.validation.cell_dsr import PERIODS_PER_YEAR

    tf = entry["timeframe"]
    got = {"registry": entry["annualisation_ppy"], "law": bars_per_year(tf), "cell_dsr": PERIODS_PER_YEAR[tf]}
    return _ok(ppy=got) if len(set(got.values())) == 1 else _fail("hash-mismatch", ppy=got)


def check_power(cell_id: str) -> dict[str, Any]:
    ps = power_status(cell_id)
    m = _LINE_REF.match(ps["ref"])
    p = resolve_ref(m["path"]) if m else None
    n_lines = len(p.read_text(encoding="utf-8").splitlines()) if p is not None and p.is_file() else 0
    exists = m is not None and 0 < int(m["line"]) <= n_lines
    detail = {"status": ps["status"], "ref": ps["ref"], "ref_line_exists": exists}
    return _ok(**detail) if ps["status"] == "COMPUTED" and exists else _fail("power-unproven", **detail)


def evidence_refs(entry: dict[str, Any]) -> list[str]:
    """Every path the entry cites: evidence[], the power ref's file, the audit file, the cited campaign manifests."""
    refs = list(entry["evidence"])
    m = _LINE_REF.match(entry["power_check"]["ref"])
    if m:
        refs.append(m["path"])
    refs.append(entry["audit_row_ref"]["file"])
    refs += _MANIFEST_REF.findall(entry["spread_artifact_hash"]["source"])
    return sorted(set(refs))


def check_evidence(entry: dict[str, Any]) -> dict[str, Any]:
    refs = evidence_refs(entry)
    missing = [r for r in refs if not resolve_ref(r).is_file()]
    detail = {"refs": len(refs), "missing": missing}
    return _fail(f"evidence-missing:{len(missing)}", **detail) if missing else _ok(**detail)


def pinned_campaign(entry: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """(manifest ref, manifest, failure reason): the ONE cited manifest whose search cell is this cell."""
    matches = []
    for ref in _MANIFEST_REF.findall(entry["spread_artifact_hash"]["source"]):
        p = resolve_ref(ref)
        if not p.is_file():
            continue
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:  # a present but unreadable manifest is a mismatch, not a missing campaign
            return None, None, "hash-mismatch"
        cfg = doc.get("run", {}).get("config", {})
        if (cfg.get("symbol"), cfg.get("timeframe"), cfg.get("dataset")) == \
                (entry["symbol"], entry["timeframe"], entry["dataset_hash16"]):
            matches.append((ref, doc))
    if len(matches) != 1:
        return None, None, "dataset-missing" if not matches else "hash-mismatch"
    return matches[0][0], matches[0][1], None


def check_rederive(entry: dict[str, Any]) -> dict[str, Any]:
    ref, manifest, why = pinned_campaign(entry)
    if why:
        return _fail(why, campaign=None, note="exactly one cited manifest must have this cell as its search cell")
    from engine.research.search.dna import DNA
    from engine.research.search.real_backtest import fast_backtest_factory, load_cell_bars_from_snapshot

    cell_id = entry["cell_id"]
    try:
        cell = load_cell_bars_from_snapshot(entry["symbol"], entry["timeframe"], entry["dataset_hash16"])
    except Exception as exc:  # the loader refuses a missing or tampered snapshot
        return _fail(_snapshot_reason(entry), campaign=ref, error=f"{type(exc).__name__}: {exc}")
    bt = fast_backtest_factory(cell)
    rows, mismatches = [], []
    for cand in manifest["candidates"]:
        r = bt(DNA(**cand["dna"]))
        got = {"sharpe": r.sharpe, "n_trades": r.n_trades, "total_return": r.total_return, "max_dd": r.max_dd,
               "blocked_max_spread": r.blocked_max_spread, "costs": dict(r.costs)}
        rec = {k: cand["metrics"][cell_id].get(k) for k in METRICS}
        same = got == rec
        rows.append({"dna_hash": cand["dna_hash"], "equal": same, **got})
        if not same:
            mismatches.append({"dna_hash": cand["dna_hash"], "recorded": rec, "rederived": got})
    detail = {"campaign": ref, "run_hash": manifest["run"].get("run_hash"), "candidates": len(rows),
              "equal": sum(r["equal"] for r in rows), "rows_sha256": hashlib.sha256(canonical(rows).encode()).hexdigest(),
              "fallback_state": dict(bt.fallback_state), "mismatches": mismatches,
              "tier": "per-cell R4(c): exact re-derivation of the pinned campaign's recorded metrics"}
    return _ok(**detail) if not mismatches and rows else _fail("hash-mismatch", **detail)


def battery_cell() -> tuple[str | None, str, str]:
    """(registry cell_id, SYMBOL, DATASET_ID) of the banked battery, read from its module constants by ast."""
    tree = ast.parse((paths.PROJECT_ROOT / BATTERY_TEST).read_text(encoding="utf-8"))
    consts = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
              if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
    sym, ds = consts["SYMBOL"], consts["DATASET_ID"]
    ids = [e["cell_id"] for e in registry.list_cells() if e["symbol"] == sym and e["dataset_hash16"] == ds]
    return (ids[0] if len(ids) == 1 else None), sym, ds


def _snapshot_reason(entry: dict[str, Any]) -> str:
    """Why the snapshot loader refused: dataset-missing when a pinned file is absent, else hash-mismatch."""
    from engine.research.data.dataset_snapshot import datasets_dir

    stem = f"{entry['symbol']}_{entry['timeframe']}_{entry['dataset_hash16']}"
    present = all((datasets_dir() / f"{stem}{ext}").is_file() for ext in (".npz", ".json"))
    return "hash-mismatch" if present else "dataset-missing"


def parse_battery_log(text: str) -> tuple[str | None, dict[str, dict[str, Any]], bool]:
    """(HEAD sha, family -> {fired, trades, mismatches}, clean '4 passed' AND EXIT=0) from a closing battery log."""
    head = re.match(r"HEAD ([0-9a-f]{40})", text)
    fams: dict[str, dict[str, Any]] = {}
    cur = None
    for line in text.splitlines():
        m = re.match(r"PARITY (\w+) ", line)
        if m:
            cur = fams.setdefault(m[1], {})
            continue
        m = re.match(r"\s+(fired|trades \(slow\)|mismatches)\s+:\s+(\S+)", line)
        if m and cur is not None:
            key = {"fired": "fired", "trades (slow)": "trades", "mismatches": "mismatches"}[m[1]]
            cur[key] = m[2] if key == "fired" else int(m[2])
    passed = (re.search(r"^4 passed(, \d+ warnings?)? in ", text, re.M) is not None
              and re.search(r"^EXIT=0\b", text, re.M) is not None)
    return (head[1] if head else None), fams, passed


def _module_file(module: str) -> str | None:
    """Repo-relative file of an engine.* / tests.* module (module.py or package/__init__.py), or None."""
    base = paths.PROJECT_ROOT.joinpath(*module.split("."))
    for p in (base.with_suffix(".py"), base / "__init__.py"):
        if p.is_file():
            return p.relative_to(paths.PROJECT_ROOT).as_posix()
    return None


def battery_closure() -> list[str]:
    """Every repo file the battery can execute: the transitive closure of engine.* / tests.* imports from the battery
    module, by ast over every Import / ImportFrom anywhere in a file (so imports inside functions count), plus each
    package's __init__.py on the way. Over-inclusion only makes the staleness check stricter."""
    seen: set[str] = set()
    todo = [BATTERY_TEST]
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        tree = ast.parse((paths.PROJECT_ROOT / rel).read_text(encoding="utf-8"))
        pkg = rel.rsplit("/", 1)[0].replace("/", ".")
        mods: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:  # a relative import, resolved against this file's package
                    parts = pkg.split(".")[: len(pkg.split(".")) - (node.level - 1)]
                    base = ".".join([*parts, base] if base else parts)
                mods += [base] + [f"{base}.{a.name}" for a in node.names]
        for mod in mods:
            if not mod.startswith(("engine", "tests")):
                continue
            parts = mod.split(".")
            for i in range(1, len(parts) + 1):  # the module and every package __init__ above it
                f = _module_file(".".join(parts[:i]))
                if f and f not in seen:
                    todo.append(f)
    return sorted(seen)


def check_battery(cell_id: str, battery_log: Path | None) -> dict[str, Any]:
    own, sym, ds = battery_cell()
    tier = "own cell (the banked battery runs on this cell)" if own == cell_id else \
        f"global tier R4(a) (the banked battery runs on {own or sym + ' ' + ds})"
    banked = {k: {"fired": f, "trades": t, "mismatches": m} for k, (f, t, m) in BANKED_BATTERY.items()}
    base = {"battery_cell": own, "tier": tier, "banked": banked}
    if battery_log is None or not battery_log.is_file():
        return _fail("battery-fail:log", **base, log=None, note="no battery log")
    text = battery_log.read_text(encoding="utf-8")
    head, fams, passed = parse_battery_log(text)
    log = {"path": _display(battery_log), "sha256": sha256_file(battery_log), "head": head}
    closure = battery_closure()
    stale = None
    if head is None:
        stale = "no HEAD line"
    else:
        diff = subprocess.run(["git", "-C", str(paths.PROJECT_ROOT), "diff", "--name-only", head, "HEAD", "--",
                               *closure], capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(paths.PROJECT_ROOT), "status", "--porcelain",
                                "--untracked-files=no", "--", *closure], capture_output=True, text=True)
        if diff.returncode != 0 or dirty.returncode != 0:
            stale = f"git failed: {(diff.stderr or dirty.stderr).strip()}"
        elif diff.stdout.strip():
            stale = f"battery inputs changed since {head[:7]}: {diff.stdout.splitlines()}"
        elif dirty.stdout.strip():  # the code that would run is not HEAD's
            stale = f"battery inputs have uncommitted changes: {dirty.stdout.splitlines()}"
    detail = {**base, "log": log, "observed": fams, "four_passed": passed, "closure_files": len(closure),
              "closure_unchanged_since_log_head": stale is None}
    if stale:
        return _fail("battery-fail:log", **detail, stale=stale)
    for fam in BANKED_BATTERY:
        if fams.get(fam) != banked[fam]:
            return _fail(f"battery-fail:{fam}", **detail)
    if set(fams) != set(BANKED_BATTERY) or not passed:
        return _fail("battery-fail:log", **detail)
    return _ok(**detail)


def check_canary(entry: dict[str, Any]) -> dict[str, Any]:
    from engine.research.search.canary import run_planted_edge_canary
    from engine.research.search.real_backtest import load_cell_bars_from_snapshot

    try:
        cell = load_cell_bars_from_snapshot(entry["symbol"], entry["timeframe"], entry["dataset_hash16"])
    except Exception as exc:
        return _fail(_snapshot_reason(entry), error=f"{type(exc).__name__}: {exc}")
    with tempfile.TemporaryDirectory(prefix="reproof-canary-") as td:
        t0 = time.perf_counter()
        res = run_planted_edge_canary(cell, fast=True, tmp_dir=td)
        elapsed = time.perf_counter() - t0
    within = elapsed <= CANARY_BUDGET_S
    print(f"  canary elapsed {elapsed:.1f}s (budget {CANARY_BUDGET_S:.0f}s)")
    detail = {"passed": bool(res.passed), "survivors": res.survivors, "exit_verdict": res.exit_verdict,
              "crashes": res.crashes, "reason": res.reason, "within_budget": within,
              "caveats": "pinned snapshot, first 1200 bars (FAST_BARS); provenance stamped MT5_LIVE (canary.py:126)"}
    return _ok(**detail) if res.passed and within else _fail("battery-fail:canary", **detail)


def _display(p: Path) -> str:
    try:
        return p.resolve().relative_to(paths.PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


def verdict(checks: dict[str, dict[str, Any]]) -> tuple[str, str | None, list[str]]:
    """('PROVEN', None, []) or ('BLOCKED', <first reason by REASON_ORDER>, [every failure])."""
    failures = [c["reason"] for c in checks.values() if not c["ok"]]
    if not failures:
        return "PROVEN", None, []
    def rank(r: str) -> tuple[int, int]:  # by precedence; among evidence-missing, the largest count wins
        base, _, n = r.partition(":")
        return REASON_ORDER.index(base), -int(n) if base == "evidence-missing" else 0

    return "BLOCKED", min(failures, key=rank), sorted(failures)


def audit_row(out_dir: Path, row: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "audit.jsonl", "a", encoding="utf-8", newline="\n") as f:
        f.write(canonical({**row, "utc": datetime.now(tz=UTC).isoformat()}) + "\n")


def _campaign_dirs(cell_id: str) -> list[str]:
    """Every GA campaign directory for the cell under <ARQVIO_HOME>/artifacts/ga, resolved now."""
    ga = paths.get_artifacts_dir() / "ga"
    return sorted(p.name for p in ga.glob(f"ga-{cell_id}-*") if p.is_dir()) if ga.is_dir() else []


def select_tier(entry: dict[str, Any]) -> str:
    """'full' (RUN@<campaign>) or 'pre-campaign' (UNRUN), from the registry entry alone; refuses a contradiction."""
    status = entry["campaign_status"]
    if status.startswith("RUN@"):
        return "full"
    if status != "UNRUN":
        raise ReproofRefused("tier-contradiction", f"campaign_status {status!r} names no tier")
    found = {"power_check": entry["power_check"]["status"], "campaign_dirs": _campaign_dirs(entry["cell_id"]),
             "cited_manifests": _MANIFEST_REF.findall(entry["spread_artifact_hash"]["source"])}
    if found["power_check"] != "BLOCKED:no-campaign" or found["campaign_dirs"] or found["cited_manifests"]:
        raise ReproofRefused("tier-contradiction", f"campaign_status UNRUN but {found}")
    return "pre-campaign"


def reproof(cell_id: str, *, out_dir: Path, battery_log: Path | None) -> dict[str, Any]:
    """Re-prove ``cell_id``; write <out_dir>/<cell_id>.proof.json and an audit row. Refuses (raises) before any
    check on an untrusted registry or a cell the registry does not hold as VALIDATED."""
    head = git_head()
    try:
        entries = registry.load()
    except (registry.RegistryError, UnicodeDecodeError) as exc:  # a byte-level tamper is refused too
        audit_row(out_dir, {"cell": cell_id, "outcome": "REFUSED", "reason": f"registry-{type(exc).__name__}",
                            "head": head})
        raise ReproofRefused(f"registry-{type(exc).__name__}", str(exc)) from exc
    entry = next((e for e in entries if e["cell_id"] == cell_id and e["status"] == "VALIDATED"), None)
    if entry is None:
        audit_row(out_dir, {"cell": cell_id, "outcome": "REFUSED", "reason": "unknown-cell", "head": head})
        raise ReproofRefused("unknown-cell", f"{cell_id!r} is not a VALIDATED registry cell; "
                                             f"the registry holds {registry.validated_cells()}")
    try:
        tier = select_tier(entry)
    except ReproofRefused as exc:
        audit_row(out_dir, {"cell": cell_id, "outcome": "REFUSED", "reason": exc.reason, "head": head})
        raise
    doc = json.loads(registry.registry_path().read_text(encoding="utf-8"))
    checks: dict[str, dict[str, Any]] = {}
    steps = (("dataset", lambda: check_dataset(entry)), ("specs_spread", lambda: check_specs_spread(entry)),
             ("audit_row", lambda: check_audit_row(entry)), ("annualisation", lambda: check_annualisation(entry)),
             ("power", lambda: check_power(cell_id)), ("evidence", lambda: check_evidence(entry)),
             ("rederive", lambda: check_rederive(entry)), ("battery", lambda: check_battery(cell_id, battery_log)),
             ("canary", lambda: check_canary(entry)))
    if tier == "pre-campaign":
        steps = tuple(s for s in steps if s[0] in PRECAMPAIGN_CHECKS)
    for name, run in steps:
        try:  # a check that cannot run records why; the proof and the audit row are always written
            checks[name] = run()
        except Exception as exc:
            checks[name] = _fail(_ON_ERROR[name], error=f"{type(exc).__name__}: {exc}")
    outcome, reason, failures = verdict(checks)
    proven = "PROVEN@" if tier == "full" else "PROVEN-PRECAMPAIGN@"
    proof = {"schema": PROOF_SCHEMA, "version": PROOF_VERSION, "cell": cell_id, "head": head,
             "registry": {"entry_hash": entry["entry_hash"], "chain_hash": doc["chain_hash"]},
             "entry": {k: entry[k] for k in ("symbol", "timeframe", "dataset_hash16", "dataset_window",
                                             "annualisation_ppy")},
             "checks": checks, "verdict": outcome, "reason": reason, "failures": failures,
             "status": proven + head[:7] if outcome == "PROVEN" else f"BLOCKED:{reason}"}
    if tier == "pre-campaign":  # the full tier's proof keeps U3's exact keys
        proof.update(tier=tier, not_applicable=NOT_APPLICABLE_PRECAMPAIGN)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{cell_id}.proof.json").write_text(canonical(proof) + "\n", encoding="utf-8", newline="\n")
    audit_row(out_dir, {"cell": cell_id, "outcome": outcome, "reason": reason, "status": proof["status"], "head": head})
    return proof
