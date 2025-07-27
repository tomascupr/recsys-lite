"""Stub implementations for optional dependencies.

This module provides lightweight Python-only stubs for heavy dependencies
(numpy, scipy, faiss, pandas) when running in CI environments.
"""

import os as _os

# CI environment detection
is_ci = _os.environ.get("CI", "").lower() == "true"

# Import all stubs
from .faiss_stubs import install_faiss_stub
from .numpy_stubs import install_numpy_stub
from .pandas_stubs import install_pandas_stub
from .scipy_stubs import install_scipy_stub

# Install all stubs at import time
install_numpy_stub()
install_scipy_stub()
install_faiss_stub()
install_pandas_stub()

__all__ = [
    "is_ci",
    "install_numpy_stub",
    "install_scipy_stub",
    "install_faiss_stub",
    "install_pandas_stub",
]
