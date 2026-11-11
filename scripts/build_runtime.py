"""Build the installed app's engine runtime: CPython itself plus the engine source, nothing compiled in between.

    <canonical venv python, CLAUDE.md "Environment"> scripts/build_runtime.py [--out build/runtime]

Why not Nuitka (the former scripts/build_engine.py): the fast backtest is numba-JIT code, and numba compiles a
function's Python bytecode at run time. A Nuitka-compiled function has no bytecode, so a compiled sidecar could not
run the fast kernel, and the app refused every campaign from it ("this sidecar is not a Python interpreter"). The
runtime built here is the same CPython 3.11 and the same packages the test suite runs on, so the installed app
computes exactly what the tests verified.

Layout of the output (Tauri ships it as the resource directory ``engine-runtime/``):

    arqvio-engine.exe            python.exe, renamed: the process name the desktop shell starts and cleans up
    arqvio-enginew.exe           pythonw.exe, renamed: power-mode worker processes start windowless from it
                                (engine/research/ga/parallel.windowless_workers)
    python311.dll, DLLs/, Lib/  the interpreter and its standard library (test suites, IDLE and Tk left out)
    Lib/site-packages/          the engine's runtime dependencies: the distributions of every third-party module
                                the engine source imports (read from its syntax tree, so an import missing from
                                pyproject still ships), closed over their own requirements and copied file by file
                                from each distribution's RECORD -- nothing else from the development environment
    app/engine/, app/configs/   the engine source and configs tracked at HEAD, plus the build stamp

The shell starts ``arqvio-engine.exe -m engine`` with PYTHONHOME=<runtime>, PYTHONPATH=app and PYTHONNOUSERSITE=1,
so neither the user's own Python setup nor a user site-packages directory can change what runs.

Exit codes: 0 built and verified; 1 a step failed (the reason is printed).
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
EXE = "arqvio-engine.exe"
# Standard-library parts the engine never imports: test suites, IDLE, Tk, the pip bootstrapper.
STDLIB_SKIP = {"test", "idlelib", "tkinter", "turtledemo", "ensurepip", "lib2to3", "site-packages", "__pycache__"}
DLLS_SKIP = re.compile(r"^(_tkinter\.pyd|tcl\d+t?\.dll|tk\d+t?\.dll|_test.*\.pyd)$", re.IGNORECASE)
BASE_FILES = ("python311.dll", "python3.dll", "vcruntime140.dll", "vcruntime140_1.dll", "LICENSE.txt")


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return out.stdout


def imported_distributions(sources: list[Path]) -> tuple[list[str], list[str]]:
    """(distributions, unresolved): the installed distributions providing the third-party top-level modules
    ``sources`` import anywhere (module level or inside functions), and the module names no distribution provides."""
    names: set[str] = set()
    for src in sources:
        for node in ast.walk(ast.parse(src.read_text(encoding="utf-8"), filename=str(src))):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    names -= set(sys.stdlib_module_names) | {"engine", "__future__"}
    provided = metadata.packages_distributions()
    owners = {n: provided[n] for n in names if n in provided}
    for n in names - set(owners):  # e.g. pywin32's win32con, importable through a .pth path entry
        owner = file_owner(n)
        if owner is not None:
            owners[n] = [owner]
    dists = sorted({d for ds in owners.values() for d in ds})
    return dists, sorted(names - set(owners))


def file_owner(module: str) -> str | None:
    """The distribution whose RECORD lists the file ``module`` imports from, or None."""
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        return None
    origin = Path(spec.origin).resolve()
    for dist in metadata.distributions():
        if any(Path(dist.locate_file(f)).resolve() == origin for f in dist.files or []):
            return dist.metadata["Name"]
    return None


def requirement_closure(roots: list[str]) -> list[metadata.Distribution]:
    """The installed distributions ``roots`` need at run time (no extras), each once, in discovery order."""
    env = default_environment()
    seen: dict[str, metadata.Distribution] = {}
    todo = [Requirement(r) for r in roots]
    while todo:
        req = todo.pop(0)
        if req.marker is not None and not req.marker.evaluate({**env, "extra": ""}):
            continue
        key = re.sub(r"[-_.]+", "-", req.name).lower()
        if key in seen:
            continue
        dist = metadata.distribution(req.name)
        seen[key] = dist
        todo.extend(Requirement(r) for r in dist.requires or [])
    return list(seen.values())


def copy_dist(dist: metadata.Distribution, site: Path) -> int:
    """Copy one distribution's installed files (its RECORD) into ``site``; files outside site-packages are skipped."""
    n = 0
    for f in dist.files or []:
        if f.parts[0] == ".." or "__pycache__" in f.parts:
            continue
        src = Path(dist.locate_file(f))
        if not src.is_file():
            continue
        dst = site / Path(*f.parts)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n += 1
    return n


def copy_tree(src: Path, dst: Path, skip_dir: set[str], skip_file: re.Pattern[str] | None = None) -> None:
    for path in src.iterdir():
        if path.is_dir():
            if path.name not in skip_dir:
                copy_tree(path, dst / path.name, skip_dir, skip_file)
        elif skip_file is None or not skip_file.match(path.name):
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst / path.name)


def tree_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


# pythonw (arqvio-enginew.exe) starts the power-mode workers with no standard streams; a write to one (a log line, a
# warning) ended a worker abruptly and the campaign died with BrokenProcessPool. Every runtime process gets a sink.
SITECUSTOMIZE = """# Written by scripts/build_runtime.py. pythonw (arqvio-enginew.exe, the power-mode workers) has no standard
# streams; a write to sys.stdout / sys.stderr = None ended a worker abruptly. They get a sink that discards.
import os
import sys

if sys.stdout is None or sys.stderr is None:
    _null = open(os.devnull, "w", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = _null
    if sys.stderr is None:
        sys.stderr = _null
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "build" / "runtime")
    a = p.parse_args(argv)
    base = Path(sys.base_prefix)
    if sys.prefix == sys.base_prefix:
        print("REFUSED: run this with the project's virtual environment python (its packages are what ships)")
        return 1
    out: Path = a.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # 1. the interpreter, renamed, and its standard library
    shutil.copy2(base / "python.exe", out / EXE)
    shutil.copy2(base / "pythonw.exe", out / EXE.replace(".exe", "w.exe"))
    for name in BASE_FILES:
        if (base / name).is_file():
            shutil.copy2(base / name, out / name)
    copy_tree(base / "DLLs", out / "DLLs", set(), DLLS_SKIP)
    copy_tree(base / "Lib", out / "Lib", STDLIB_SKIP)
    (out / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
    (out / "Lib" / "site-packages" / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")

    # 2. the runtime dependencies, nothing else from the development environment
    tracked = [line for line in git("ls-files", "engine", "configs").splitlines() if line]
    roots, unresolved = imported_distributions([ROOT / rel for rel in tracked if rel.endswith(".py")])
    if unresolved:
        print(f"REFUSED: the engine imports modules no installed distribution provides: {unresolved}")
        return 1
    site = out / "Lib" / "site-packages"
    dists = requirement_closure(roots)
    files = sum(copy_dist(d, site) for d in dists)
    names = sorted(d.metadata["Name"] for d in dists)
    print(f"site-packages: {len(dists)} distributions, {files} files: {', '.join(names)}")

    # 3. the engine and its configs exactly as tracked at HEAD, plus the build stamp
    app = out / "app"
    for rel in tracked:
        src = ROOT / rel
        if src.is_file():
            (app / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, app / rel)
    commit = git("rev-parse", "HEAD").strip()
    if git("status", "--porcelain", "--", "engine", "configs").strip():
        commit += "+dirty"  # a tree with uncommitted edits is not the commit it names
    utc = datetime.now(tz=UTC).isoformat()
    (app / "engine" / "core" / "_build_stamp.py").write_text(
        '"""Generated by scripts/build_runtime.py. Do not edit, do not commit."""\n\n'
        f'BUILD_COMMIT = "{commit}"\nBUILD_UTC = "{utc}"\n', encoding="utf-8")
    print(f"app: {len(tracked)} tracked files, commit {commit}")

    # 4. verify: the runtime starts, imports the engine and JIT-compiles the fast kernel on its own
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("PYTHON")}
    env |= {"PYTHONHOME": str(out), "PYTHONPATH": str(app), "PYTHONNOUSERSITE": "1", "PYTHONIOENCODING": "utf-8"}
    smoke = str(ROOT / "scripts" / "runtime_pool_smoke.py")
    checks = [
        [str(out / EXE), smoke],
        [str(out / EXE), "-m", "engine", "--print-build"],
        [str(out / EXE), "-c", "import sys, numba, numpy as np; from engine.research.search import real_backtest; "
                               "from engine.research.backtest.kernel import _pairwise_sum; "
                               "print('numba', numba.__version__, 'prefix', sys.prefix, "
                               "'jit', _pairwise_sum(np.arange(10.0), 0, 10))"],
    ]
    for cmd in checks:
        r = subprocess.run(cmd, cwd=app, env=env, capture_output=True, text=True, timeout=600)
        print(f"$ {' '.join(cmd[1:])}\n{r.stdout.strip()}")
        if r.returncode != 0:
            print(f"verification failed (rc={r.returncode}):\n{r.stderr.strip()[-2000:]}")
            return 1
        if cmd[-1] == smoke and ("pool 12" not in r.stdout or "enginew.exe" not in r.stdout):
            print("verification failed: a pool of windowless workers did not finish (pool 12 from the twin expected)")
            return 1
        if "-c" in cmd and "jit 45.0" not in r.stdout:
            print("verification failed: the fast kernel did not JIT-compile and sum 0..9 to 45.0")
            return 1
        if "--print-build" in cmd and commit not in r.stdout:
            print("verification failed: the runtime does not report this build's commit")
            return 1
    print(f"runtime ready: {out} ({tree_size(out) / 1048576:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
