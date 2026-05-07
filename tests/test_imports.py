"""
Basic import tests for nn project.
"""

import importlib
import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_import_hooks():
    """
    Test importing hooks module.
    """

    module = importlib.import_module("hooks")

    assert module is not None


def test_project_root_exists():
    """
    Ensure project root exists.
    """

    assert os.path.exists(PROJECT_ROOT)
