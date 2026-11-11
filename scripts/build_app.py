"""Build the ARQVIO installers: the engine runtime, then the desktop app with it bundled.

    <canonical venv python, CLAUDE.md "Environment"> scripts/build_app.py [--skip-runtime]

1. scripts/build_runtime.py   -> build/runtime (CPython + the engine's dependencies + the engine at HEAD), verified:
                                 it starts, reports its commit and JIT-compiles the fast kernel
2. npm run tauri build        -> apps/desktop/src-tauri/target/release/bundle/{nsis,msi}/ARQVIO_<version>_x64*
                                 (the runtime ships as the resource directory engine-runtime/)
3. prints every installer with its size and sha256, so a release names exactly the bytes it ships

Exit codes: 0 built; 1 a step failed (its own output says why).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "apps" / "desktop"
BUNDLE = DESKTOP / "src-tauri" / "target" / "release" / "bundle"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-runtime", action="store_true", help="reuse build/runtime (UI-only rebuild)")
    a = p.parse_args(argv)
    if not a.skip_runtime:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_runtime.py")], cwd=ROOT)
        if r.returncode != 0:
            return 1
    if not (ROOT / "build" / "runtime" / "arqvio-engine.exe").is_file():
        print("no build/runtime: run without --skip-runtime first")
        return 1
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if npm is None:
        print("npm not found on PATH")
        return 1
    r = subprocess.run([npm, "run", "tauri", "build"], cwd=DESKTOP, env={**os.environ, "CI": "true"})
    if r.returncode != 0:
        print(f"tauri build failed: {r.returncode}")
        return 1
    found = sorted([*BUNDLE.glob("nsis/*.exe"), *BUNDLE.glob("msi/*.msi")])
    if not found:
        print(f"no installer under {BUNDLE}")
        return 1
    for f in found:
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        print(f"{f}  {f.stat().st_size / 1048576:.1f} MB  sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
