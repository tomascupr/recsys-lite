"""Testing utilities and MagicMock patches for RecSys-Lite.

This module contains testing helpers and patches to make MagicMock work
better with the CLI testing framework.
"""

from .mocks import (
    install_magicmock_patches,
    patch_get_type_hints,
    patch_test_cli_module,
    patch_typer_testing,
)

# Install all patches at import time
install_magicmock_patches()
patch_get_type_hints()
patch_typer_testing()

__all__ = [
    "install_magicmock_patches",
    "patch_test_cli_module",
    "patch_get_type_hints",
    "patch_typer_testing",
]
