"""Tests for the __init__.py file in kaira.models.fec package."""

import importlib
import sys


def test_import_all():
    """Test that all components can be imported from the package."""
    # Clear any previous import if it exists
    if "kaira.models.fec" in sys.modules:
        del sys.modules["kaira.models.fec"]

    # Import the module
    import kaira.models.fec

    # Check that the module has all expected submodules
    assert hasattr(kaira.models.fec, "algebra")
    assert hasattr(kaira.models.fec, "utils")
    assert hasattr(kaira.models.fec, "encoders")
    assert hasattr(kaira.models.fec, "decoders")

    # Verify that __all__ is as expected
    assert kaira.models.fec.__all__ == ["algebra", "encoders", "decoders", "utils"]


def test_reimport():
    """Test that the module can be reimported without errors."""
    importlib.reload(importlib.import_module("kaira.models.fec"))

    # Import should complete without errors

    assert True
