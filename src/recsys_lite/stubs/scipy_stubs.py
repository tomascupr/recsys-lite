"""Minimal scipy.sparse stub implementation.

Enough for the unit-tests that construct ``lil_matrix`` and convert it to CSR.
The matrices are represented by plain Python dictionaries keyed by ``(row, col)``
tuples; only a subset of the real API is implemented.
"""

import os as _os
import sys as _sys
import types as _types
from typing import Any, Dict, Tuple

# Use stubs when the CI environment variable is set
is_ci = _os.environ.get("CI", "").lower() == "true"


def install_scipy_stub() -> None:  # pragma: no cover
    """Install scipy stub if in CI environment."""
    # Only install stubs in CI environment
    if not is_ci:
        return

    if "scipy" in _sys.modules:
        return

    _sp_mod = _types.ModuleType("scipy")
    _sp_sparse_mod = _types.ModuleType("scipy.sparse")

    class _LilMatrix:  # Very small subset
        def __init__(self, shape: Tuple[int, int], dtype: Any = None) -> None:
            self._shape = shape
            self._data: Dict[Tuple[int, int], float] = {}

        def __setitem__(self, idx: Tuple[int, int], value: float) -> None:
            self._data[idx] = value

        def tocsr(self) -> "_CsrMatrix":  # noqa: D401
            return _CsrMatrix(self._shape, self._data.copy())

    class _CsrMatrix:
        def __init__(self, shape: Tuple[int, int], data: Dict[Tuple[int, int], float]) -> None:
            self.shape = shape
            self._data = data

    _sp_sparse_mod.lil_matrix = _LilMatrix  # type: ignore[attr-defined]
    _sp_sparse_mod.csr_matrix = _CsrMatrix  # type: ignore[attr-defined]

    _sp_mod.sparse = _sp_sparse_mod  # type: ignore[attr-defined]

    _sys.modules["scipy"] = _sp_mod
    _sys.modules["scipy.sparse"] = _sp_sparse_mod
