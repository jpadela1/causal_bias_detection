"""
Environment diagnostic for the causal_bias_audit project.

Run this FIRST in IDE, BEFORE any of the main_*.py scripts:

    Right-click check_environment.py in the Project pane > Run

If you see ModuleNotFoundError or "missing packages" elsewhere, this script
isolates the cause. It does four things in order:

    1. Prints which Python interpreter is running this code
    2. Tries to import every required package and reports versions.
    3. Probes the causal-learn submodules we actually use, since some of them
       have non-obvious internal dependencies (pydot + system graphviz).
    4. Suggests a copy-pasteable pip command for whatever's missing.


"""

from __future__ import annotations

import importlib
import platform
import shutil
import sys
import sysconfig
from pathlib import Path

# --------------------------------------------------------------------------- #
# Required top-level packages and the minimum version we expect.
# (None means "any version that imports".)
# --------------------------------------------------------------------------- #
REQUIRED = [
    ("numpy", "1.23"),
    ("pandas", "1.5"),
    ("scipy", "1.10"),
    ("sklearn", "1.2"),          # imported as sklearn, package is scikit-learn
    ("matplotlib", "3.6"),
    ("seaborn", "0.12"),
    ("networkx", "2.8"),
    ("tqdm", "4.65"),
    ("statsmodels", "0.14"),     # required by causal-learn internally
    ("pydot", "2.0"),             # required by causal-learn for graph rendering
    ("causallearn", "0.1.4.0"),   # imported as causallearn, package is causal-learn
]

# Pip names (only differ where pip name != import name)
PIP_NAME = {
    "sklearn": "scikit-learn",
    "causallearn": "causal-learn",
}

# causal-learn submodules the project actually touches. If the top-level import
# works but these don't, you have a version or partial-install problem.
CAUSAL_LEARN_SUBMODULES = [
    "causallearn.search.ConstraintBased.PC",
    "causallearn.search.ConstraintBased.FCI",
    "causallearn.search.ScoreBased.GES",
    "causallearn.search.PermutationBased.GRaSP",
    "causallearn.search.FCMBased.lingam",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _v_tuple(s: str) -> tuple[int, ...]:
    out = []
    for chunk in s.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def _meets_min(actual: str | None, minimum: str | None) -> bool:
    if minimum is None or actual is None:
        return True
    try:
        return _v_tuple(actual) >= _v_tuple(minimum)
    except Exception:
        return True  # don't fail closed on weird versions


def _interpreter_info() -> None:
    print("=" * 72)
    print("PYTHON INTERPRETER")
    print("=" * 72)
    print(f"  sys.executable : {sys.executable}")
    print(f"  sys.version    : {sys.version.split()[0]}")
    print(f"  platform       : {platform.platform()}")
    print(f"  prefix         : {sys.prefix}")

    in_venv = sys.prefix != sys.base_prefix
    print(f"  in virtualenv? : {in_venv}")
    if not in_venv:
        print()
        print("  WARNING: you are NOT running inside a virtualenv.")
        print("  In PyCharm this usually means the project interpreter is set")
        print("  to your system Python. See INSTALL.md > 'Step 2: interpreter'")

    site_packages = sysconfig.get_paths().get("purelib", "<unknown>")
    print(f"  site-packages  : {site_packages}")
    print()


def _system_tools() -> None:
    print("=" * 72)
    print("SYSTEM TOOLS (needed by pydot for graph rendering)")
    print("=" * 72)
    dot = shutil.which("dot")
    if dot:
        print(f"  graphviz 'dot' : {dot}  OK")
    else:
        print("  graphviz 'dot' : NOT FOUND ON PATH")
        print()
        print("  pydot will install but rendering DAGs may fail. Fix:")
        print("    macOS  : brew install graphviz")
        print("    Ubuntu : sudo apt install graphviz")
        print("    Windows: https://graphviz.org/download/  (then add bin/ to PATH)")
    print()


def _check_packages() -> list[tuple[str, str]]:
    """Returns list of (import_name, reason) for failed packages."""
    print("=" * 72)
    print("REQUIRED PACKAGES")
    print("=" * 72)
    failed: list[tuple[str, str]] = []

    for import_name, min_ver in REQUIRED:
        try:
            mod = importlib.import_module(import_name)
        except ImportError as exc:
            print(f"  [MISSING ] {import_name:15s}  -> {exc}")
            failed.append((import_name, "missing"))
            continue
        except Exception as exc:
            # E.g. NumPy ABI mismatch raises non-ImportError on import.
            print(f"  [BROKEN  ] {import_name:15s}  -> {type(exc).__name__}: {exc}")
            failed.append((import_name, "broken"))
            continue

        version = getattr(mod, "__version__", "?")
        if _meets_min(version, min_ver):
            print(f"  [OK      ] {import_name:15s}  v{version}")
        else:
            print(f"  [TOO OLD ] {import_name:15s}  v{version}  (need >= {min_ver})")
            failed.append((import_name, "too_old"))
    print()
    return failed


def _check_causal_learn_submodules() -> list[str]:
    print("=" * 72)
    print("CAUSAL-LEARN SUBMODULES")
    print("=" * 72)
    failed: list[str] = []
    for sub in CAUSAL_LEARN_SUBMODULES:
        try:
            importlib.import_module(sub)
            print(f"  [OK      ] {sub}")
        except Exception as exc:
            print(f"  [FAILED  ] {sub}")
            print(f"             -> {type(exc).__name__}: {exc}")
            failed.append(sub)
    print()
    return failed


def _check_numpy_compat_shim() -> bool:
    """Verify that numpy_compat applies its patches correctly. Returns True
    if the shim is healthy, False otherwise."""
    print("=" * 72)
    print("NUMPY 2.x COMPATIBILITY SHIM")
    print("=" * 72)
    try:
        import numpy_compat
    except ImportError:
        print("  [MISSING ] numpy_compat.py not found in project root")
        print("             This file is required for GES and GRaSP to work")
        print("             on NumPy 2.x. Re-download it from the project.")
        print()
        return False

    applied = numpy_compat.patches_applied()
    import numpy as _np
    print(f"  NumPy version  : {_np.__version__}")
    print(f"  patches active : {len(applied)}")
    if applied:
        for p in applied[:5]:
            print(f"    - {p}")
        if len(applied) > 5:
            print(f"    - ... and {len(applied) - 5} more")

    # Smoke-test the two attributes that GES and GRaSP need.
    ok = True
    if not hasattr(_np, "mat"):
        print("  [BROKEN  ] np.mat is missing -- GES/GRaSP will fail")
        ok = False
    if not hasattr(_np, "matlib"):
        print("  [BROKEN  ] np.matlib is missing -- GES/GRaSP will fail")
        ok = False
    if ok:
        print("  np.mat and np.matlib both available -- GES/GRaSP imports should work")

    # Also verify that the post-import scoring patch is callable.
    # We do not actually invoke it here (that requires causal-learn to be
    # installed), but we confirm the function exists so the user knows
    # the fix for the scoring TypeError is wired up.
    if hasattr(numpy_compat, "patch_causal_learn_scoring"):
        print("  patch_causal_learn_scoring() available -- BIC scoring fix ready")
    else:
        print("  [BROKEN  ] patch_causal_learn_scoring() missing -- update numpy_compat.py")
        ok = False
    print()
    return ok



def _suggest_fix(failed_pkgs: list[tuple[str, str]], failed_subs: list[str]) -> None:
    print("=" * 72)
    print("SUGGESTED FIX")
    print("=" * 72)

    if not failed_pkgs and not failed_subs:
        print("  Environment looks healthy. You can run main_synthetic.py.")
        return

    if failed_pkgs:
        pip_names = [PIP_NAME.get(name, name) for name, _ in failed_pkgs]
        print("  Run THIS in PyCharm's built-in terminal (Alt+F12), NOT a")
        print("  separate system terminal -- the built-in terminal auto-")
        print("  activates the project venv:")
        print()
        print("      pip install --upgrade " + " ".join(pip_names))
        print()
        print("  Then re-run check_environment.py to confirm.")

    if failed_subs and not failed_pkgs:
        print("  causal-learn imported but its submodules failed. This usually")
        print("  means a version mismatch or a NumPy 2.x ABI break. Try:")
        print()
        print("      pip install --upgrade --force-reinstall causal-learn numpy")
    print()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> int:
    print()
    print("Diagnostic for", Path(__file__).resolve().parent.name)
    print()
    _interpreter_info()
    _system_tools()
    failed_pkgs = _check_packages()
    shim_ok = _check_numpy_compat_shim() if not any(
        name == "numpy" for name, _ in failed_pkgs
    ) else True
    failed_subs = _check_causal_learn_submodules() if not any(
        name == "causallearn" for name, _ in failed_pkgs
    ) else []
    _suggest_fix(failed_pkgs, failed_subs)
    return 0 if (not failed_pkgs and not failed_subs and shim_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
