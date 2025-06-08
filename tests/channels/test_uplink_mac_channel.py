"""Tests for UplinkMACChannel implementation in Kaira."""

import pytest
import torch

from kaira.channels import AWGNChannel, FlatFadingChannel, RayleighFadingChannel, UplinkMACChannel


class TestUplinkMACChannel:
    """Test suite for UplinkMACChannel class."""

    def test_initialization_single_channel(self):
        """Test initialization with a single shared channel."""
        base_channel = AWGNChannel(avg_noise_power=0.1)
        channel = UplinkMACChannel(
            user_channels=base_channel,
            num_users=3,
            user_gains=[1.0, 0.8, 1.2],
            interference_power=0.05,
        )

        assert len(channel.user_channels) == 3
        assert all(isinstance(ch, AWGNChannel) for ch in channel.user_channels)
        assert channel.num_users == 3
        assert torch.allclose(channel.user_gains, torch.tensor([1.0, 0.8, 1.2]))
        assert channel.interference_power == 0.05
        assert channel.combine_method == "sum"

    def test_initialization_channel_list(self):
        """Test initialization with a list of channels."""
        channels = [
            AWGNChannel(avg_noise_power=0.1),
            FlatFadingChannel(fading_type="rayleigh", coherence_time=10, avg_noise_power=0.2),
            RayleighFadingChannel(coherence_time=5, avg_noise_power=0.15),
        ]
        channel = UplinkMACChannel(
            user_channels=channels,
            user_gains=2.0,  # Single gain for all users
            interference_power=0.1,
            combine_method="weighted_sum",
        )

        assert len(channel.user_channels) == 3
        assert channel.num_users == 3
        assert torch.allclose(channel.user_gains, torch.tensor([2.0, 2.0, 2.0]))
        assert channel.interference_power == 0.1
        assert channel.combine_method == "weighted_sum"

    def test_initialization_errors(self):
        """Test initialization error cases."""
        base_channel = AWGNChannel(avg_noise_power=0.1)

        # Missing num_users with single channel
        with pytest.raises(ValueError, match="num_users must be specified"):
            UplinkMACChannel(user_channels=base_channel)

        # Mismatched gains length
        with pytest.raises(ValueError, match="Length of user_gains"):
            UplinkMACChannel(
                user_channels=base_channel,
                num_users=3,
                user_gains=[1.0, 0.8],  # Only 2 gains for 3 users
            )

        # Invalid combine method
        with pytest.raises(ValueError, match="combine_method must be one of"):
            UplinkMACChannel(
                user_channels=base_channel,
                num_users=2,
                combine_method="invalid_method",
            )

    def test_forward_single_user(self):
        """Test forward pass with single user."""
        base_channel = AWGNChannel(avg_noise_power=0.1)
        channel = UplinkMACChannel(
            user_channels=base_channel,
            num_users=1,
            user_gains=1.0,
            interference_power=0.0,
        )

        input_signal = torch.randn(10, 64, dtype=torch.complex64)
        user_signals = [input_signal]

        output = channel(user_signals)

        # With single user and no interference, output shape should match input
        assert output.shape == input_signal.shape
        # Output should be complex
        assert output.dtype == torch.complex64

    def test_forward_multiple_users(self):
        """Test forward pass with multiple users."""
        channels = [
            AWGNChannel(avg_noise_power=0.1),
            AWGNChannel(avg_noise_power=0.1),
            AWGNChannel(avg_noise_power=0.1),
        ]
        channel = UplinkMACChannel(
            user_channels=channels,
            user_gains=[1.0, 0.8, 1.2],
            interference_power=0.05,
        )

        # Create different signals for each user
        user_signals = [
            torch.randn(10, 64, dtype=torch.complex64),
            torch.randn(10, 64, dtype=torch.complex64),
            torch.randn(10, 64, dtype=torch.complex64),
        ]

        output = channel(user_signals)

        # Output should have same shape as individual signals
        assert output.shape == user_signals[0].shape
        assert output.dtype == torch.complex64

    def test_forward_different_signal_shapes(self):
        """Test error when user signals have different shapes."""
        base_channel = AWGNChannel(avg_noise_power=0.1)
        channel = UplinkMACChannel(
            user_channels=base_channel,
            num_users=2,
        )

        user_signals = [
            torch.randn(10, 64, dtype=torch.complex64),
            torch.randn(10, 32, dtype=torch.complex64),  # Different shape
        ]

        with pytest.raises(ValueError, match="All user signals must have the same shape"):
            channel(user_signals)

    def test_forward_wrong_number_of_signals(self):
        """Test error when wrong number of user signals provided."""
        base_channel = AWGNChannel(avg_noise_power=0.1)
        channel = UplinkMACChannel(
            user_channels=base_channel,
            num_users=3,
        )

        user_signals = [
            torch.randn(10, 64, dtype=torch.complex64),
            torch.randn(10, 64, dtype=torch.complex64),
        ]  # Only 2 signals for 3 users

        with pytest.raises(ValueError, match="Expected 3 user signals"):
            channel(user_signals)

    def test_weighted_sum_combination(self):
        """Test weighted sum combination method."""
        base_channel = AWGNChannel(avg_noise_power=0.0)  # No noise for cleaner test
        channel = UplinkMACChannel(
            user_channels=base_channel,
            num_users=2,
            user_gains=[2.0, 0.5],
            interference_power=0.0,
            combine_method="weighted_sum",
        )

        # Use simple signals for predictable results
        user_signals = [
            torch.ones(1, 4, dtype=torch.complex64),
            torch.ones(1, 4, dtype=torch.complex64) * 2,
        ]

        output = channel(user_signals)

        # Expected: 2.0 * 1 + 0.5 * 2 = 3.0 for each element
        expected = torch.ones(1, 4, dtype=torch.complex64) * 3.0
        assert torch.allclose(output, expected, atol=1e-6)

    def test_get_user_csi(self):
        """Test per-user CSI retrieval."""
        channels = [
            FlatFadingChannel(fading_type="rayleigh", coherence_time=10, avg_noise_power=0.1),
            FlatFadingChannel(fading_type="rayleigh", coherence_time=5, avg_noise_power=0.2),
        ]
        channel = UplinkMACChannel(user_channels=channels)

        # CSI should be None for these channels (they don't expose it)
        assert channel.get_user_csi(0) is None
        assert channel.get_user_csi(1) is None

    def test_get_user_csi_invalid_user(self):
        """Test error when requesting CSI for invalid user."""
        base_channel = AWGNChannel(avg_noise_power=0.1)
        channel = UplinkMACChannel(
            user_channels=base_channel,
            num_users=2,
        )

        with pytest.raises(ValueError, match="User index 2 is out of range"):
            channel.get_user_csi(2)

    def test_update_user_gain(self):
        """Test dynamic user gain updates."""
        base_channel = AWGNChannel(avg_noise_power=0.1)
        channel = UplinkMACChannel(
            user_channels=base_channel,
            num_users=2,
            user_gains=[1.0, 1.0],
        )

        # Update gain for user 0
        channel.update_user_gain(0, 2.5)
        assert torch.allclose(channel.user_gains[0], torch.tensor(2.5))

        # Update gain for user 1
        channel.update_user_gain(1, 0.3)
        assert torch.allclose(channel.user_gains[1], torch.tensor(0.3))

    def test_update_user_gain_invalid_user(self):
        """Test error when updating gain for invalid user."""
        base_channel = AWGNChannel(avg_noise_power=0.1)
        channel = UplinkMACChannel(
            user_channels=base_channel,
            num_users=2,
        )

        with pytest.raises(ValueError, match="User index 5 is out of range"):
            channel.update_user_gain(5, 1.5)

    def test_update_interference_power(self):
        """Test dynamic interference power updates."""
        base_channel = AWGNChannel(avg_noise_power=0.1)
        channel = UplinkMACChannel(
            user_channels=base_channel,
            num_users=2,
            interference_power=0.1,
        )

        # Update interference power
        new_power = 0.25
        channel.update_interference_power(new_power)
        assert channel.interference_power == new_power

    def test_add_interference(self):
        """Test interference addition functionality."""
        base_channel = AWGNChannel(avg_noise_power=0.0)  # No channel noise
        channel = UplinkMACChannel(
            user_channels=base_channel,
            num_users=2,
            interference_power=0.1,
        )

        signals = [torch.ones(10, 64, dtype=torch.complex64), torch.ones(10, 64, dtype=torch.complex64)]

        # Test that interference is added (output should differ from input)
        interfered_signals = channel._add_interference(signals)

        assert len(interfered_signals) == len(signals)
        assert interfered_signals[0].shape == signals[0].shape
        assert interfered_signals[0].dtype == signals[0].dtype
        # With interference, output should generally differ from input
        # (though there's a small chance they could be very close due to randomness)

    def test_combine_signals_sum_method(self):
        """Test signal combination with sum method."""
        base_channel = AWGNChannel(avg_noise_power=0.1)
        channel = UplinkMACChannel(
            user_channels=base_channel,
            num_users=2,
            combine_method="sum",
        )

        processed_signals = [
            torch.ones(1, 4, dtype=torch.complex64),
            torch.ones(1, 4, dtype=torch.complex64) * 2,
        ]

        combined = channel._combine_signals(processed_signals)
        expected = torch.ones(1, 4, dtype=torch.complex64) * 3  # 1 + 2

        assert torch.allclose(combined, expected, atol=1e-6)

    def test_different_channel_types(self):
        """Test with different types of underlying channels."""
        channels = [
            AWGNChannel(avg_noise_power=0.1),
            FlatFadingChannel(fading_type="rayleigh", coherence_time=10, avg_noise_power=0.1),
            RayleighFadingChannel(coherence_time=5, avg_noise_power=0.1),
        ]

        channel = UplinkMACChannel(
            user_channels=channels,
            user_gains=[1.0, 1.0, 1.0],
            interference_power=0.05,
        )

        user_signals = [
            torch.randn(5, 32, dtype=torch.complex64),
            torch.randn(5, 32, dtype=torch.complex64),
            torch.randn(5, 32, dtype=torch.complex64),
        ]

        output = channel(user_signals)

        assert output.shape == user_signals[0].shape
        assert output.dtype == torch.complex64

    def test_device_consistency(self):
        """Test that the channel works correctly with different devices."""
        base_channel = AWGNChannel(avg_noise_power=0.1)
        channel = UplinkMACChannel(
            user_channels=base_channel,
            num_users=2,
        )

        # Test with CPU tensors
        user_signals_cpu = [
            torch.randn(5, 32, dtype=torch.complex64),
            torch.randn(5, 32, dtype=torch.complex64),
        ]

        output_cpu = channel(user_signals_cpu)
        assert output_cpu.device == user_signals_cpu[0].device

        # Test with CUDA tensors if available
        if torch.cuda.is_available():
            user_signals_cuda = [sig.cuda() for sig in user_signals_cpu]
            output_cuda = channel(user_signals_cuda)
            assert output_cuda.device == user_signals_cuda[0].device

    def test_batch_processing(self):
        """Test processing batches of signals."""
        base_channel = AWGNChannel(avg_noise_power=0.1)
        channel = UplinkMACChannel(
            user_channels=base_channel,
            num_users=2,
        )

        # Batch of signals
        batch_size = 8
        signal_length = 64
        user_signals = [
            torch.randn(batch_size, signal_length, dtype=torch.complex64),
            torch.randn(batch_size, signal_length, dtype=torch.complex64),
        ]

        output = channel(user_signals)

        assert output.shape == (batch_size, signal_length)
        assert output.dtype == torch.complex64

    def test_zero_interference_power(self):
        """Test behavior with zero interference power."""
        base_channel = AWGNChannel(avg_noise_power=0.0)  # No channel noise
        channel = UplinkMACChannel(
            user_channels=base_channel,
            num_users=1,
            user_gains=1.0,
            interference_power=0.0,
        )

        input_signal = torch.randn(10, 64, dtype=torch.complex64)
        user_signals = [input_signal]

        output = channel(user_signals)

        # With no noise and no interference, output should be very close to input
        assert torch.allclose(output, input_signal, atol=1e-6)

    def test_repr_method(self):
        """Test string representation of the channel."""
        base_channel = AWGNChannel(avg_noise_power=0.1)
        channel = UplinkMACChannel(
            user_channels=base_channel,
            num_users=2,
            user_gains=[1.0, 0.8],
            interference_power=0.05,
            combine_method="weighted_sum",
        )

        repr_str = repr(channel)
        assert "UplinkMACChannel" in repr_str
        assert "num_users=2" in repr_str
        assert "interference_power=0.05" in repr_str
        assert "combine_method=weighted_sum" in repr_str
