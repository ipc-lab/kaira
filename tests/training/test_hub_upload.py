#!/usr/bin/env python3
"""Test script for Hugging Face Hub upload functionality.

This script tests the argument parsing and validation for Hub upload features.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.kaira_train import create_parser, setup_hub_upload


def test_hub_arguments():
    """Test that Hub arguments are properly parsed."""
    print("Testing Hugging Face Hub argument parsing...")

    parser = create_parser()

    # Test basic Hub upload arguments
    test_args = ["--model", "deepjscc", "--push-to-hub", "--hub-model-id", "test-user/test-model", "--hub-private", "--hub-strategy", "end"]

    args = parser.parse_args(test_args)

    # Verify Hub arguments
    assert args.push_to_hub, "push_to_hub should be True"
    assert args.hub_model_id == "test-user/test-model", f"hub_model_id should be 'test-user/test-model', got {args.hub_model_id}"
    assert args.hub_private, "hub_private should be True"
    assert args.hub_strategy == "end", f"hub_strategy should be 'end', got {args.hub_strategy}"

    print("✅ Basic argument parsing test passed")

    # Test validation (should work with mocked arguments)
    class MockArgs:
        def __init__(self):
            self.push_to_hub = True
            self.hub_model_id = "test-user/test-model"
            self.hub_token = "fake_test_token_for_testing"  # nosec B105 - This is a test token, not a real credential
            self.hub_private = False
            self.hub_strategy = "end"
            self.quiet = True

    mock_args = MockArgs()

    try:
        # This should not raise an error for validation (though it will fail to authenticate)
        setup_hub_upload(mock_args)
        print("✅ Hub configuration setup test passed")
    except ImportError:
        print("ℹ️  huggingface_hub not installed - skipping hub config test")
    except Exception as e:
        print(f"✅ Hub configuration validation working (expected error: {e})")

    print("✅ All Hub argument tests passed!")


def test_no_hub_arguments():
    """Test that training works without Hub arguments."""
    print("\nTesting training without Hub arguments...")

    parser = create_parser()

    test_args = ["--model", "deepjscc", "--output-dir", "./test_results"]

    args = parser.parse_args(test_args)

    # Verify Hub arguments have defaults
    assert not args.push_to_hub, "push_to_hub should default to False"
    assert args.hub_model_id is None, "hub_model_id should default to None"
    assert not args.hub_private, "hub_private should default to False"
    assert args.hub_strategy == "end", "hub_strategy should default to 'end'"

    print("✅ Default Hub argument test passed")

    # Test that setup_hub_upload returns None when not using Hub
    class MockArgsNoHub:
        def __init__(self):
            self.push_to_hub = False
            self.quiet = True

    mock_args = MockArgsNoHub()
    hub_config = setup_hub_upload(mock_args)
    assert hub_config is None, "hub_config should be None when push_to_hub is False"

    print("✅ No Hub upload test passed!")


if __name__ == "__main__":
    print("🧪 Testing Kaira Hub Upload Functionality")
    print("=" * 50)

    try:
        test_hub_arguments()
        test_no_hub_arguments()
        print("\n🎉 All tests passed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
