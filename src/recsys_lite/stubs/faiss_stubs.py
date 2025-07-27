"""Minimal faiss stub implementation.

Purely Python implementation that fulfils the public surface required by the
test-suite. No actual ANN search is performed – the ``search`` method merely
returns dummy distances and indices.
"""

import os as _os
import sys as _sys
import types as _types
from typing import Any, Tuple

# Use stubs when the CI environment variable is set
is_ci = _os.environ.get("CI", "").lower() == "true"


def install_faiss_stub() -> None:  # pragma: no cover
    """Install faiss stub if in CI environment."""
    # Only install stubs in CI environment
    if not is_ci:
        return

    if "faiss" in _sys.modules:
        return

    _np = _sys.modules.get("numpy")

    class _FakeIndex:
        def __init__(self, dim: int):
            self.d = dim  # Faiss stores dimensionality in `.d`
            self.nprobe = 10

        def add(self, vectors: Any) -> None:  # noqa: D401
            self._vectors = vectors  # noqa: SLF001 – simple field

        def train(self, vectors: Any) -> None:  # noqa: D401 – no‑op
            pass

        def search(self, query: Any, k: int) -> Tuple[Any, Any]:  # noqa: D401
            n = len(query) if isinstance(query, list) else 1
            # Distances all zeros, indices sequential – sufficient for asserts
            dists = _np.zeros((n, k)) if _np else [[0] * k for _ in range(n)]
            idxs = _np.zeros((n, k), dtype=int) if _np else [[0] * k for _ in range(n)]
            return dists, idxs

    _faiss = _types.ModuleType("faiss")

    # Metric constants
    _faiss.METRIC_INNER_PRODUCT = 0
    _faiss.METRIC_L2 = 1

    # Make metric constants directly accessible from the module
    _faiss.METRIC_INNER_PRODUCT = 0
    _faiss.METRIC_L2 = 1

    # Expose _FakeIndex as the faiss.Index class for type annotation compatibility
    _faiss.Index = _FakeIndex

    # Factory helpers -------------------------------------------------------
    _faiss.IndexFlatL2 = lambda dim: _FakeIndex(dim)  # type: ignore[attr-defined]
    _faiss.IndexFlatIP = lambda dim: _FakeIndex(dim)  # type: ignore[attr-defined]

    def _index_ivf_flat(quantizer: Any, dim: int, nlist: int, metric: Any) -> "_FakeIndex":  # noqa: D401
        return _FakeIndex(dim)

    _faiss.IndexIVFFlat = _index_ivf_flat  # type: ignore[attr-defined]
    _faiss.IndexHNSWFlat = lambda dim, m: _FakeIndex(dim)  # type: ignore[attr-defined]

    def _normalize_L2(vecs: Any) -> Any:  # noqa: D401 – no‑op
        return vecs

    _faiss.normalize_L2 = _normalize_L2  # type: ignore[attr-defined]

    # Add functions for reading/writing indices
    def _write_index(index: Any, path: str) -> None:  # noqa: D401 - stub
        # Just a stub - does nothing in test mode
        pass

    def _read_index(path: str) -> "_FakeIndex":  # noqa: D401 - stub
        # Return a fake index
        return _FakeIndex(100)  # Default 100-dim index

    _faiss.write_index = _write_index
    _faiss.read_index = _read_index

    _sys.modules["faiss"] = _faiss
