"""Configuration for pytest."""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path automatically for all tests
project_root = Path(__file__).parent.parent  # Get the project root directory
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))


# Create markers for tests
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "requires_deps: mark test as requiring specific dependencies")
    config.addinivalue_line("markers", "slow: mark test as slow running")


# Environment detection helpers
def is_ci_environment():
    """Check if tests are running in a CI environment."""
    return os.environ.get("CI", "false").lower() == "true"
