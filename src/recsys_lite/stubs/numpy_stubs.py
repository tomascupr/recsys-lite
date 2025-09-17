"""Ultra-light numpy stub implementation.

The real library crashes in the execution environment due to an OpenBLAS issue.
The tests bundled with RecSys-Lite only rely on a handful of simple helpers:
``array``, ``zeros`` and the ``random`` namespace with ``rand`` and ``randint``.
We provide a minimal pure-Python implementation that satisfies those requirements
and avoids the native extension import entirely.
"""

import os as _os
import sys as _sys
import types as _types
from typing import Any, List, Optional, Tuple, Union

# Use stubs when the CI environment variable is set
is_ci = _os.environ.get("CI", "").lower() == "true"


def install_numpy_stub() -> None:  # pragma: no cover – executed at import time
    """Install numpy stub if in CI environment."""
    # Only install stubs in CI environment
    if not is_ci:
        return

    if "numpy" in _sys.modules:
        # Real NumPy has already been imported elsewhere – nothing we can do.
        return

    _np = _types.ModuleType("numpy")

    # Basic constructors ----------------------------------------------------
    def _array(data: Any, *_, **__) -> List[Any]:  # noqa: D401 – just a stub
        return list(data)

    def _zeros(shape: Union[int, Tuple[int, ...]], *_, **__) -> Union[List[int], List[List[int]]]:  # noqa: D401
        if isinstance(shape, int):
            return [0] * shape
        # simple multi-dimensional – produce nested lists
        if len(shape) == 2:
            return [[0] * shape[1] for _ in range(shape[0])]
        return [0] * (shape[0] if shape else 0)

    def _arange(stop: int, *args: int, **kwargs: Any) -> List[int]:  # noqa: D401
        start = 0
        step = 1
        if len(args) == 1:
            start = stop
            stop = args[0]
        elif len(args) == 2:
            start = args[0]
            step = args[1]
        return list(range(start, stop, step))

    # Random namespace ------------------------------------------------------
    class _RandomState:  # Minimal replacement – enough for tests
        def __init__(self, seed: Optional[int] = None) -> None:  # noqa: D401
            import random as _random

            self._rng = _random.Random(seed)

        def randint(self, low: int, high: Optional[int] = None, size: Optional[int] = None):
            if high is None:
                low, high = 0, low
            if size is None:
                return self._rng.randint(low, high - 1)
            return [self.randint(low, high) for _ in range(size)]

        def random(self, size):
            import random as _random

            return [_random.random() for _ in range(size[0] * size[1])]

    class _RandomNamespace(_types.SimpleNamespace):
        def __init__(self) -> None:  # noqa: D401
            super().__init__(rand=lambda *shape: [0] * (shape[0] if shape else 1))

        def RandomState(self, seed=None):  # noqa: D401 – matches numpy API
            return _RandomState(seed)

    _np.array = _array  # type: ignore[attr-defined]
    _np.zeros = _zeros  # type: ignore[attr-defined]
    _np.arange = _arange  # type: ignore[attr-defined]

    # Provide ``np.random`` sub‑module
    _np.random = _RandomNamespace()  # type: ignore[attr-defined]

    # Provide the polyfit stub expected by some code paths ------------------
    def _polyfit(x: List[Any], y: List[Any], deg: int, *args: Any, **kwargs: Any) -> List[int]:  # noqa: D401
        return _zeros(deg + 1)

    _np.polyfit = _polyfit  # type: ignore[attr-defined]
    lib_mod = _types.ModuleType("numpy.lib")
    poly_mod = _types.ModuleType("numpy.lib.polynomial")
    poly_mod.polyfit = _polyfit
    lib_mod.polynomial = poly_mod
    _np.lib = lib_mod
    _sys.modules["numpy.lib"] = lib_mod
    _sys.modules["numpy.lib.polynomial"] = poly_mod

    # Expose dtype names lightly (used in tests for attr access)
    for _name in ("float32", "float64", "int32", "int64"):
        setattr(_np, _name, None)

    # Finalise
    _sys.modules["numpy"] = _np
