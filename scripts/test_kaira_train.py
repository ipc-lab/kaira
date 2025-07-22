#!/usr/bin/env python3
"""Test script for kaira-train console script."""

import subprocess  # nosec B404
import sys
import tempfile


def run_command(cmd, check=True):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        raise


def test_kaira_train_help():
    """Test that kaira-train --help works."""
    print("\n=== Testing kaira-train --help ===")
    result = run_command([sys.executable, "-m", "scripts.kaira_train", "--help"])
    assert "Kaira Training CLI" in result.stdout
    print("✓ Help command works")


def test_kaira_train_list_models():
    """Test that kaira-train --list-models works."""
    print("\n=== Testing kaira-train --list-models ===")
    result = run_command([sys.executable, "-m", "scripts.kaira_train", "--list-models"])
    assert "Available Models:" in result.stdout
    print("✓ List models command works")


def test_kaira_train_invalid_model():
    """Test that kaira-train fails gracefully with invalid model."""
    print("\n=== Testing kaira-train with invalid model ===")
    with tempfile.TemporaryDirectory() as temp_dir:
        result = run_command([sys.executable, "-m", "scripts.kaira_train", "--model", "nonexistent_model", "--output-dir", temp_dir, "--epochs", "1"], check=False)
        assert result.returncode != 0
        print("✓ Invalid model properly rejected")


def test_script_imports():
    """Test that the script can be imported without errors."""
    print("\n=== Testing script imports ===")
    try:
        # Try to import the script as a module
        import importlib.util

        spec = importlib.util.find_spec("scripts.kaira_train")
        if spec is not None:
            print("✓ Script can be found and imported")
        else:
            raise ImportError("scripts.kaira_train module not found")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        raise


def main():
    """Run all tests."""
    print("Testing kaira-train console script...")

    try:
        test_script_imports()
        test_kaira_train_help()
        test_kaira_train_list_models()
        test_kaira_train_invalid_model()

        print("\n=== All tests passed! ===")
        return True

    except Exception as e:
        print(f"\n=== Test failed: {e} ===")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
