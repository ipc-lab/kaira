"""Tests for channel registry and special channel implementations."""

import pytest
import torch

from kaira.channels import IdentityChannel, LambdaChannel, PerfectChannel
from kaira.channels.base import BaseChannel
from kaira.channels.registry import ChannelRegistry


class TestChannelRegistry:
    """Test suite for ChannelRegistry."""

    def test_register_channel(self):
        """Test registering a new channel."""

        # Create a test channel class
        class TestChannel(BaseChannel):
            def __init__(self, param=None):
                super().__init__()
                self.param = param

            def forward(self, x):
                return x

        # Register the channel
        ChannelRegistry.register("test_channel", TestChannel)

        # Verify registration
        assert "test_channel" in ChannelRegistry._channels
        assert ChannelRegistry._channels["test_channel"] is TestChannel

    def test_get_channel(self):
        """Test getting a registered channel."""

        # Register a test channel
        class TestChannel(BaseChannel):
            def forward(self, x):
                return x

        ChannelRegistry.register("test_get_channel", TestChannel)

        # Get the channel
        channel_class = ChannelRegistry.get("test_get_channel")
        assert channel_class is TestChannel

    def test_nonexistent_channel(self):
        """Test getting a non-existent channel."""
        with pytest.raises(KeyError):
            ChannelRegistry.get("nonexistent_channel")

    def test_create_channel(self):
        """Test creating a channel instance."""

        # Register a test channel
        class TestChannel(BaseChannel):
            def __init__(self, param=None):
                super().__init__()
                self.param = param

            def forward(self, x):
                return x

        ChannelRegistry.register("test_create_channel", TestChannel)

        # Create an instance
        instance = ChannelRegistry.create("test_create_channel", param=42)

        # Verify instance
        assert isinstance(instance, TestChannel)
        assert instance.param == 42

    def test_decorator_registration(self):
        """Test channel registration via decorator."""

        # Define and register a channel using the decorator
        @ChannelRegistry.register_channel("decorated_channel")
        class DecoratedChannel(BaseChannel):
            def forward(self, x):
                return x

        # Verify registration
        assert "decorated_channel" in ChannelRegistry._channels
        assert ChannelRegistry._channels["decorated_channel"] is DecoratedChannel

        # Test with default name (lowercase class name)
        @ChannelRegistry.register_channel()
        class AutoNamedChannel(BaseChannel):
            def forward(self, x):
                return x

        assert "autonamedchannel" in ChannelRegistry._channels

    def test_list_channels(self):
        """Test listing all registered channels."""
        # Clear existing channels (just for this test)
        original_channels = ChannelRegistry._channels.copy()
        ChannelRegistry._channels = {}

        try:
            # Register some test channels
            class TestChannel1(BaseChannel):
                def forward(self, x):
                    return x

            class TestChannel2(BaseChannel):
                def forward(self, x):
                    return x

            ChannelRegistry.register("test_channel1", TestChannel1)
            ChannelRegistry.register("test_channel2", TestChannel2)

            # Get the list
            channel_list = ChannelRegistry.list_channels()

            # Verify list
            assert "test_channel1" in channel_list
            assert "test_channel2" in channel_list
            assert len(channel_list) == 2

        finally:
            # Restore original channels
            ChannelRegistry._channels = original_channels


class TestPerfectChannel:
    """Test suite for PerfectChannel."""

    def test_perfect_channel_real(self, random_tensor):
        """Test PerfectChannel with real input."""
        channel = PerfectChannel()
        output = channel(random_tensor)

        # Output should be identical to input
        assert torch.equal(output, random_tensor)

    def test_perfect_channel_complex(self, complex_tensor):
        """Test PerfectChannel with complex input."""
        channel = PerfectChannel()
        output = channel(complex_tensor)

        # Output should be identical to input
        assert torch.equal(output, complex_tensor)

    def test_identity_aliases(self):
        """Test that IdentityChannel is an alias for PerfectChannel."""
        assert IdentityChannel is PerfectChannel

        # Create instances
        perfect = PerfectChannel()
        identity = IdentityChannel()

        # Both should be instances of PerfectChannel
        assert isinstance(perfect, PerfectChannel)
        assert isinstance(identity, PerfectChannel)

        # Test basic functionality
        x = torch.randn(10)
        assert torch.equal(perfect(x), x)
        assert torch.equal(identity(x), x)


class TestLambdaChannel:
    """Test suite for LambdaChannel."""

    def test_initialization(self):
        """Test initialization with a function."""

        # Define a test function
        def test_fn(x):
            return 2 * x

        channel = LambdaChannel(fn=test_fn)
        assert channel.fn is test_fn

    def test_real_input(self, random_tensor):
        """Test with real input."""

        # Define a doubling function
        def double_fn(x):
            return 2 * x

        channel = LambdaChannel(fn=double_fn)
        output = channel(random_tensor)

        # Check shape preservation
        assert output.shape == random_tensor.shape

        # Check output is doubled input
        expected = 2 * random_tensor
        assert torch.allclose(output, expected)

    def test_complex_input(self, complex_tensor):
        """Test with complex input."""

        # Define a function that adds a constant complex value
        def add_complex_fn(x):
            return x + torch.complex(torch.tensor(1.0), torch.tensor(1.0))

        channel = LambdaChannel(fn=add_complex_fn)
        output = channel(complex_tensor)

        # Check shape preservation
        assert output.shape == complex_tensor.shape

        # Check output matches expected transformation
        expected = complex_tensor + torch.complex(torch.tensor(1.0), torch.tensor(1.0))
        assert torch.allclose(output, expected)

    def test_stateful_function(self):
        """Test with a stateful function."""
        # Create a channel with a closure that tracks state
        counter = [0]

        def stateful_fn(x):
            counter[0] += 1
            return x + counter[0]

        channel = LambdaChannel(fn=stateful_fn)

        # Apply multiple times
        x = torch.ones(5)
        y1 = channel(x)  # Should add 1
        y2 = channel(x)  # Should add 2

        # Check that state is maintained
        assert torch.all(y1 == 2)
        assert torch.all(y2 == 3)

    def test_with_additional_args(self, random_tensor):
        """Test LambdaChannel with additional arguments."""

        # Define a function with additional parameters
        def parameterized_fn(x, scale=1.0, offset=0.0):
            return scale * x + offset

        channel = LambdaChannel(fn=parameterized_fn)

        # Call with additional arguments
        output = channel(random_tensor, scale=2.0, offset=1.0)

        # Check output matches expected transformation
        expected = 2.0 * random_tensor + 1.0
        assert torch.allclose(output, expected)
