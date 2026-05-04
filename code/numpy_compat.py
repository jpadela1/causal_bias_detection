"""
numpy_compat.py
===============
Restore NumPy 1.x attributes that NumPy 2.x removed, so that older
libraries (notably causal-learn's GES and GRaSP score functions) keep
working without downgrading NumPy.

This is a programmatic fix: nothing in your installed NumPy is modified
on disk. We attach the missing attributes to the live ``numpy`` module
object at import time. Any subsequent ``import numpy as np`` in the same
process picks up the patched module.

What we patch and why
---------------------
1. ``np.mat`` and ``np.matlib`` -- removed in NumPy 2.0.
   causal-learn issue #208 (https://github.com/py-why/causal-learn/issues/208)
   triggers immediately on ``from causallearn.search.ScoreBased.GES import ges``
   because the GES score machinery wraps inputs in ``np.mat(...)``. Likewise
   GRaSP routes through the same score functions, so it inherits the bug.

2. ``np.float`` / ``np.int`` / ``np.bool`` / ``np.object`` / ``np.complex``
   / ``np.long`` / ``np.unicode`` / ``np.str`` -- aliases removed in
   NumPy 1.24. causal-learn still has straggling references (e.g. dtype
   declarations in older score-function code paths).

3. ``np.product`` / ``np.cumproduct`` / ``np.alltrue`` / ``np.sometrue``
   / ``np.row_stack`` / ``np.in1d`` -- removed in NumPy 2.0. We map them
   to their replacements (``np.prod``, ``np.cumprod``, ``np.all``,
   ``np.any``, ``np.vstack``, ``np.isin``).

4. ``np.NaN`` / ``np.Inf`` / ``np.NAN`` / ``np.PINF`` / ``np.NINF``
   / ``np.Infinity`` -- removed/renamed in NumPy 2.0.

How to use
----------
Import this module BEFORE importing anything from ``causallearn``::

    import numpy_compat   # noqa: F401  (apply patches)
    from causallearn.search.ScoreBased.GES import ges

The order matters because some causal-learn modules call ``np.mat`` during
their own import (module-level code), so the shim must be active before
those imports happen.

The patches are idempotent and safe on NumPy 1.x: each one is wrapped in
a ``hasattr`` check so we never overwrite an attribute that already exists
correctly.
"""
from __future__ import annotations

import warnings

import numpy as np


def _patch_attribute(name: str, value) -> bool:
    """Attach ``value`` to ``numpy`` as ``name`` if it's not already there.

    Returns True if a patch was applied, False if numpy already had it.
    NumPy 2.x emits FutureWarning for ``hasattr(np, 'object')`` / ``np.str``
    (the names are reserved for future rebinding), so we silence those
    here -- our intent is exactly to detect their absence and patch.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        already_present = hasattr(np, name)
    if not already_present:
        setattr(np, name, value)
        return True
    return False


_PATCHES_APPLIED: list[str] = []


def apply_patches(verbose: bool = False) -> list[str]:
    """Apply all NumPy 2.x compatibility patches. Idempotent."""
    applied: list[str] = []

    # --- 1. np.mat / np.matlib -------------------------------------------
    # np.asmatrix is still present in NumPy 2.x and behaves identically to
    # np.mat for the call patterns causal-learn uses (wrapping ndarrays).
    if _patch_attribute("mat", np.asmatrix):
        applied.append("np.mat -> np.asmatrix")

    # numpy.matlib was removed but is sometimes accessed as np.matlib.zeros
    # etc. Provide a thin shim object with the methods causal-learn uses.
    if not hasattr(np, "matlib"):
        class _MatlibShim:
            """Minimal replacement for the removed numpy.matlib namespace."""
            @staticmethod
            def zeros(shape, dtype=float):
                return np.asmatrix(np.zeros(shape, dtype=dtype))

            @staticmethod
            def ones(shape, dtype=float):
                return np.asmatrix(np.ones(shape, dtype=dtype))

            @staticmethod
            def eye(n, M=None, k=0, dtype=float):
                return np.asmatrix(np.eye(n, M=M, k=k, dtype=dtype))

            @staticmethod
            def identity(n, dtype=None):
                return np.asmatrix(np.identity(n, dtype=dtype))

            @staticmethod
            def empty(shape, dtype=float):
                return np.asmatrix(np.empty(shape, dtype=dtype))

            @staticmethod
            def repmat(a, m, n):
                # numpy.matlib.repmat -> np.tile equivalent for matrices
                return np.asmatrix(np.tile(np.asarray(a), (m, n)))

        np.matlib = _MatlibShim()  # type: ignore[attr-defined]
        applied.append("np.matlib -> shim namespace")

    # --- 2. Deprecated dtype aliases (removed in NumPy 1.24) -------------
    # These were aliases to Python builtins. causal-learn never relied on
    # the ndarray-deprecation distinction, so the simple identity mapping
    # is correct.
    type_aliases = {
        "float":   float,
        "int":     int,
        "bool":    bool,
        "object":  object,
        "complex": complex,
        "long":    int,
        "unicode": str,
        "str":     str,
    }
    for name, target in type_aliases.items():
        if _patch_attribute(name, target):
            applied.append(f"np.{name} -> Python {target.__name__}")

    # --- 3. Removed-but-renamed functions --------------------------------
    function_aliases = {
        "product":    np.prod,
        "cumproduct": np.cumprod,
        "alltrue":    np.all,
        "sometrue":   np.any,
        "row_stack":  np.vstack,
        "in1d":       np.isin,
    }
    for name, target in function_aliases.items():
        if _patch_attribute(name, target):
            applied.append(f"np.{name} -> np.{target.__name__}")

    # --- 4. Removed/renamed numeric constants ----------------------------
    constant_aliases = {
        "NaN":      np.nan,
        "NAN":      np.nan,
        "Inf":      np.inf,
        "Infinity": np.inf,
        "PINF":     np.inf,
        "NINF":    -np.inf,
    }
    for name, value in constant_aliases.items():
        if _patch_attribute(name, value):
            applied.append(f"np.{name} -> {value}")

    # --- 5. Scalar-conversion compatibility for size-1 matrices ----------
    # NumPy 2.x tightened the rules on float()/int() for ndim>0 arrays.
    # causal-learn's BIC scoring sometimes does float(matrix_expr) where
    # the matrix is shape (1,1). On older NumPy this worked silently; on
    # NumPy 2.x it can raise TypeError("only size-1 arrays can be
    # converted to Python scalars") if the intermediate is an ndarray
    # rather than np.matrix.
    #
    # We patch np.matrix.__float__ / __int__ / __complex__ to extract the
    # single element via .item() when size == 1. np.matrix is a Python-
    # level class (unlike np.ndarray) so monkey-patching is safe.
    try:
        _original_matrix_float = np.matrix.__float__
        _original_matrix_int = np.matrix.__int__

        def _matrix_float(self):
            if self.size == 1:
                return float(self.item())
            return _original_matrix_float(self)

        def _matrix_int(self):
            if self.size == 1:
                return int(self.item())
            return _original_matrix_int(self)

        # Only install once: subsequent imports shouldn't re-wrap.
        if not getattr(np.matrix.__float__, "_compat_wrapped", False):
            _matrix_float._compat_wrapped = True
            _matrix_int._compat_wrapped = True
            np.matrix.__float__ = _matrix_float
            np.matrix.__int__ = _matrix_int
            applied.append("np.matrix.__float__ -> size-1 .item() extraction")
            applied.append("np.matrix.__int__ -> size-1 .item() extraction")
    except Exception as exc:  # pragma: no cover -- defensive
        warnings.warn(f"numpy_compat: could not patch np.matrix scalar conversion: {exc}")

    # --- 6. Suppress the noisy "size-1 array to scalar" deprecation ------
    # NumPy 1.25+ emits DeprecationWarning when float(arr) is called on a
    # 1-element array. causal-learn's BIC scoring does this in inner loops,
    # which floods the console without affecting correctness. Filter it.
    warnings.filterwarnings(
        "ignore",
        message=r".*Conversion of an array with ndim > 0 to a scalar.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*np\.bool8.*deprecated.*",
        category=DeprecationWarning,
    )

    if verbose and applied:
        print("[numpy_compat] applied patches:")
        for item in applied:
            print(f"  - {item}")
    elif verbose:
        print("[numpy_compat] NumPy already has all compat attributes, no patches needed.")

    _PATCHES_APPLIED.extend(p for p in applied if p not in _PATCHES_APPLIED)
    return applied


# Apply patches at import time. Subsequent imports of this module are
# free no-ops because every patch is hasattr-guarded.
apply_patches(verbose=False)


def patches_applied() -> list[str]:
    """Return the list of patches that were applied at import time.

    Useful for debugging: if causal-learn is still failing, check whether
    the patch you expected actually fired.
    """
    return list(_PATCHES_APPLIED)


# --------------------------------------------------------------------------- #
# Causal-learn scoring patch
# --------------------------------------------------------------------------- #
# This is a SEPARATE patch from the NumPy import-time shim above. It must be
# applied AFTER causal-learn has been imported, because it replaces the
# ``float`` name inside the already-loaded causallearn.score.LocalScoreFunction
# module. Call patch_causal_learn_scoring() once after importing causal-learn
# but before invoking GES or GRaSP.
# --------------------------------------------------------------------------- #
def _make_lenient_float():
    """Build a ``float()`` replacement that handles size-1 ndarrays of any dim.

    NumPy 2.x's ``float(arr)`` only accepts truly 0-dimensional arrays;
    ``float(np.array([[1.0]]))`` (shape (1,1)) now raises
    ``TypeError: only 0-dimensional arrays can be converted to Python scalars``.

    causal-learn's BIC scoring computes ``yX @ XX_inv @ yX.T`` which yields
    a (1,1) ndarray, then calls ``float(...)`` on it. The expression below
    returns the single element via ``.item()`` for any size-1 array,
    regardless of dimensionality, and falls through to the builtin for
    everything else.
    """
    _builtin_float = float

    def lenient_float(x=0.0, /):
        if isinstance(x, np.ndarray) and x.size == 1:
            return _builtin_float(x.item())
        if isinstance(x, np.matrix) and x.size == 1:
            return _builtin_float(x.item())
        return _builtin_float(x)

    lenient_float.__name__ = "lenient_float"
    lenient_float._compat_lenient = True
    return lenient_float


def _make_lenient_int():
    """Same idea as _make_lenient_float, but for ``int(arr)`` calls."""
    _builtin_int = int

    def lenient_int(x=0, /):
        if isinstance(x, np.ndarray) and x.size == 1:
            return _builtin_int(x.item())
        if isinstance(x, np.matrix) and x.size == 1:
            return _builtin_int(x.item())
        return _builtin_int(x)

    lenient_int.__name__ = "lenient_int"
    lenient_int._compat_lenient = True
    return lenient_int


# Modules whose namespace we inject lenient float/int into. Each call to
# float(x) inside these modules will then use the lenient version (Python's
# LEGB scoping checks module globals before builtins).
_CAUSAL_LEARN_TARGETS = (
    "causallearn.score.LocalScoreFunction",
    "causallearn.score.LocalScoreFunctionClass",
)


def patch_causal_learn_scoring(verbose: bool = False) -> list[str]:
    """Inject NumPy 2.x-tolerant float() / int() into causal-learn scoring.

    Must be called AFTER ``import causallearn.*`` has happened. The patch
    is idempotent: a sentinel attribute on each patched module ensures we
    don't re-wrap on repeated calls.

    Returns the list of module names that were patched on this call (empty
    if all targets were already patched, or if causal-learn is not
    installed).
    """
    lenient_float = _make_lenient_float()
    lenient_int = _make_lenient_int()
    patched: list[str] = []

    for mod_name in _CAUSAL_LEARN_TARGETS:
        try:
            mod = __import__(mod_name, fromlist=["*"])
        except ImportError:
            continue
        if getattr(mod, "_compat_scoring_patched", False):
            continue
        # Inject as module-level names. Python's name resolution prefers
        # module globals over builtins, so any ``float(x)`` / ``int(x)``
        # call defined in this module now uses our lenient version.
        mod.float = lenient_float
        mod.int = lenient_int
        mod._compat_scoring_patched = True
        patched.append(mod_name)

    if patched:
        for name in patched:
            entry = f"{name}.float -> lenient_float (NumPy 2.x size-1 array handler)"
            if entry not in _PATCHES_APPLIED:
                _PATCHES_APPLIED.append(entry)

    if verbose:
        if patched:
            print("[numpy_compat] patched causal-learn scoring modules:")
            for name in patched:
                print(f"  - {name}")
        else:
            print("[numpy_compat] causal-learn scoring already patched (no-op).")

    return patched
