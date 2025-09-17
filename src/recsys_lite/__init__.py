# ---------------------------------------------------------------------------
# Environment safety knobs – Keep BLAS thread pools low to avoid heavy CPU
# usage and sporadic segmentation faults observed in some CI environments.
# ---------------------------------------------------------------------------

import os as _os

# Set these only if the user has not configured them already.
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")

# Use stubs when the CI environment variable is set
# This allows for simpler testing and development without heavy dependencies
is_ci = _os.environ.get("CI", "").lower() == "true"

# ---------------------------------------------------------------------------
# Install stubs and testing patches
# ---------------------------------------------------------------------------

# Import and install stub implementations for CI environments
from .stubs import (
    install_numpy_stub as install_numpy_stub,
    install_scipy_stub as install_scipy_stub,
    install_faiss_stub as install_faiss_stub,
    install_pandas_stub as install_pandas_stub,
)

# Import and install testing patches
from .testing import (
    install_magicmock_patches as install_magicmock_patches,
    patch_test_cli_module,
    patch_get_type_hints as patch_get_type_hints,
    patch_typer_testing as patch_typer_testing,
)

# Apply delayed test CLI module patching
patch_test_cli_module()

# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------

"""RecSys-Lite: Lightweight recommendation system for small e-commerce shops."""

__version__ = "0.3.2"

# Export core models for backward compatibility
from .models import ALSModel as ALSModel, EASEModel as EASEModel
