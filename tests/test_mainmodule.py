# tests/test_kaira.py
from kaira import __version__


def test_version():
    """Test that the version is defined."""
    assert __version__ is not None
