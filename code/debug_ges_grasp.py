"""
debug_ges_grasp.py
==================
Standalone diagnostic that runs ONLY GES and GRaSP on a small synthetic
dataset and writes any traceback both to the console and to a file at:

    debug_output/ges_grasp_traceback.txt

Run this in PyCharm: right-click in the editor -> Run 'debug_ges_grasp'.
When it finishes (success or failure), open
``debug_output/ges_grasp_traceback.txt`` to read or copy the result.

This sidesteps any console-truncation, stderr-redirection, or Run-window
buffering issues that may be hiding the [FAIL] block from main_synthetic.py.
"""
from __future__ import annotations

import os
import sys
import traceback
from io import StringIO
from pathlib import Path

# IMPORTANT: numpy_compat must be imported before causal-learn or any of
# our wrappers that import it. causal_discovery handles this for us, but
# we re-import explicitly here for clarity.
import numpy_compat  # noqa: F401

from synthetic_data import generate_loan_data
from causal_discovery import run_ges, run_grasp


OUT_DIR = Path("debug_output")
OUT_FILE = OUT_DIR / "ges_grasp_traceback.txt"


def _run_one(name: str, fn, data, log: StringIO) -> None:
    """Run a single algorithm and capture success / failure to ``log``."""
    banner = "=" * 72
    log.write(f"\n{banner}\n")
    log.write(f"  RUNNING: {name}\n")
    log.write(f"{banner}\n")
    print(f"\n{banner}")
    print(f"  RUNNING: {name}")
    print(banner)

    try:
        result = fn(data)
    except Exception as exc:
        # Capture the traceback as a string so we can write it to both
        # stdout and the log file.
        tb_str = traceback.format_exc()
        msg = (
            f"\n[FAIL] {name} raised {type(exc).__name__}: {exc}\n"
            f"\nFull traceback:\n{tb_str}\n"
        )
        print(msg)
        log.write(msg)
        return

    n_dir = len(result.directed_edges)
    n_und = len(result.undirected_edges)
    n_bi = len(result.bidirected_edges)
    msg = (
        f"\n[OK] {name} succeeded.\n"
        f"     {n_dir} directed, {n_und} undirected, {n_bi} bidirected edges.\n"
    )
    print(msg)
    log.write(msg)


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)

    # Capture environment context first -- helps me diagnose if it's a
    # version-specific issue.
    log = StringIO()
    log.write("=" * 72 + "\n")
    log.write("ENVIRONMENT\n")
    log.write("=" * 72 + "\n")
    log.write(f"  Python:    {sys.version.split()[0]}\n")
    log.write(f"  Executable: {sys.executable}\n")

    try:
        import numpy as np
        log.write(f"  NumPy:     {np.__version__}\n")
    except ImportError:
        log.write("  NumPy:     NOT INSTALLED\n")

    try:
        import causallearn
        log.write(f"  causal-learn: {getattr(causallearn, '__version__', 'unknown')}\n")
    except ImportError:
        log.write("  causal-learn: NOT INSTALLED\n")

    try:
        applied = numpy_compat.patches_applied()
        log.write(f"  numpy_compat patches applied: {len(applied)}\n")
        for p in applied:
            log.write(f"    - {p}\n")
    except Exception as exc:
        log.write(f"  numpy_compat: FAILED ({exc})\n")

    print(log.getvalue())

    # Generate a tiny dataset -- 500 samples is plenty to reproduce
    # the BIC scoring path that GES/GRaSP exercise.
    print("Generating 500-sample synthetic loan dataset...")
    data = generate_loan_data(n=500, beta=-0.15, seed=42)
    log.write(f"\nData shape: {data.shape}\n")
    log.write(f"Columns: {list(data.columns)}\n")

    # Run GES, then GRaSP. Each is independent -- a failure in one
    # does not skip the other.
    _run_one("GES", run_ges, data, log)
    _run_one("GRaSP", run_grasp, data, log)

    # Write the full log to a file so it's easy to copy/paste.
    OUT_FILE.write_text(log.getvalue(), encoding="utf-8")
    print()
    print("=" * 72)
    print(f"  Full log written to: {OUT_FILE.resolve()}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
