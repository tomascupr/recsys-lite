"""Tiny pandas stub implementation.

Enough to keep import statements alive in the subset of tests that are executed
when the *CI* environment variable is set. We do *not* aim for full DataFrame
functionality, only for the constructor and two I/O helpers used in the
ingestion tests, both of which become skipped in CI mode anyway. The
implementation therefore acts as a graceful placeholder.
"""

import os as _os
import sys as _sys
import types as _types
from pathlib import Path as _Path

# Use stubs when the CI environment variable is set
is_ci = _os.environ.get("CI", "").lower() == "true"


def install_pandas_stub() -> None:  # pragma: no cover
    """Install pandas stub if in CI environment."""
    # Only install stubs in CI environment
    if not is_ci:
        return

    if "pandas" in _sys.modules:
        return

    _pd = _types.ModuleType("pandas")

    class _DataFrame:  # noqa: D401 – minimal placeholder
        def __init__(self, data=None, *args, **kwargs):
            self._data = data

        # Very naive implementations – just enough for tests to call without crashing
        def to_parquet(self, path, *_, **__):  # noqa: D401
            _p = _Path(path)
            _p.write_text("parquet_stub")

        def to_csv(self, path, *_, **__):  # noqa: D401
            _p = _Path(path)
            _p.write_text("csv_stub")

    _pd.DataFrame = _DataFrame  # type: ignore[attr-defined]

    # Return empty Series stub for completeness
    class _Series:  # noqa: D401
        pass

    _pd.Series = _Series  # type: ignore[attr-defined]

    _sys.modules["pandas"] = _pd
