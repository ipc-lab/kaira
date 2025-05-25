"""Tests for MLP encoder and decoder components."""

import torch
import torch.nn as nn

from kaira.models.components.mlp import MLPDecoder, MLPEncoder


class TestMLPEncoder:
    """Test MLPEncoder class."""

    def test_init_default(self):
        """Test MLPEncoder initialization with default parameters."""
        encoder = MLPEncoder(in_features=10, out_features=5)
        assert hasattr(encoder, "model")
        assert isinstance(encoder.model, nn.Sequential)

        # Check that model has the right structure
        layers = list(encoder.model.children())
        assert len(layers) >= 2  # At least input layer and output layer

    def test_init_custom_hidden_dims(self):
        """Test MLPEncoder initialization with custom hidden dimensions."""
        hidden_dims = [32, 16, 8]
        encoder = MLPEncoder(in_features=10, out_features=5, hidden_dims=hidden_dims)

        # Count linear layers (should be len(hidden_dims) + 1 for output)
        linear_layers = [layer for layer in encoder.model if isinstance(layer, nn.Linear)]
        assert len(linear_layers) == len(hidden_dims) + 1

    def test_init_custom_activation(self):
        """Test MLPEncoder initialization with custom activation."""
        custom_activation = nn.Tanh()
        encoder = MLPEncoder(in_features=10, out_features=5, activation=custom_activation)

        # Check that Tanh activations are used
        tanh_layers = [layer for layer in encoder.model if isinstance(layer, nn.Tanh)]
        assert len(tanh_layers) > 0

    def test_init_output_activation(self):
        """Test MLPEncoder initialization with output activation."""
        output_activation = nn.Sigmoid()
        encoder = MLPEncoder(in_features=10, out_features=5, output_activation=output_activation)

        # Check that last layer is the output activation
        last_layer = list(encoder.model.children())[-1]
        assert isinstance(last_layer, nn.Sigmoid)

    def test_init_no_hidden_dims(self):
        """Test MLPEncoder initialization without hidden dimensions."""
        encoder = MLPEncoder(in_features=10, out_features=5, hidden_dims=[])

        # Should only have one linear layer (input to output)
        linear_layers = [layer for layer in encoder.model if isinstance(layer, nn.Linear)]
        assert len(linear_layers) == 1
        assert linear_layers[0].in_features == 10
        assert linear_layers[0].out_features == 5

    def test_forward_basic(self):
        """Test forward pass with basic configuration."""
        encoder = MLPEncoder(in_features=10, out_features=5)
        x = torch.randn(32, 10)  # Batch size 32

        output = encoder(x)
        assert output.shape == (32, 5)

    def test_forward_custom_hidden(self):
        """Test forward pass with custom hidden dimensions."""
        encoder = MLPEncoder(in_features=20, out_features=8, hidden_dims=[64, 32, 16])
        x = torch.randn(16, 20)

        output = encoder(x)
        assert output.shape == (16, 8)

    def test_forward_single_sample(self):
        """Test forward pass with single sample."""
        encoder = MLPEncoder(in_features=5, out_features=3)
        x = torch.randn(1, 5)

        output = encoder(x)
        assert output.shape == (1, 3)

    def test_forward_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        encoder = MLPEncoder(in_features=10, out_features=5)

        for batch_size in [1, 8, 32, 128]:
            x = torch.randn(batch_size, 10)
            output = encoder(x)
            assert output.shape == (batch_size, 5)

    def test_forward_with_output_activation(self):
        """Test forward pass with output activation."""
        encoder = MLPEncoder(in_features=10, out_features=5, output_activation=nn.Sigmoid())
        x = torch.randn(32, 10)

        output = encoder(x)
        assert output.shape == (32, 5)
        # Output should be in [0, 1] range due to sigmoid
        assert torch.all(output >= 0) and torch.all(output <= 1)

    def test_forward_with_kwargs(self):
        """Test forward pass with additional kwargs."""
        encoder = MLPEncoder(in_features=10, out_features=5)
        x = torch.randn(32, 10)

        # Should handle extra kwargs gracefully
        output = encoder(x, some_extra_arg=42)
        assert output.shape == (32, 5)

    def test_parameter_count(self):
        """Test that parameter count is reasonable."""
        encoder = MLPEncoder(in_features=10, out_features=5, hidden_dims=[20])

        total_params = sum(p.numel() for p in encoder.parameters())
        # Expected: (10*20 + 20) + (20*5 + 5) = 220 + 105 = 325
        expected_params = (10 * 20 + 20) + (20 * 5 + 5)
        assert total_params == expected_params

    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        encoder = MLPEncoder(in_features=10, out_features=5)
        x = torch.randn(32, 10, requires_grad=True)

        output = encoder(x)
        loss = output.sum()
        loss.backward()

        # Check that input gradients exist
        assert x.grad is not None
        # Check that model parameters have gradients
        for param in encoder.parameters():
            assert param.grad is not None


class TestMLPDecoder:
    """Test MLPDecoder class."""

    def test_init_default(self):
        """Test MLPDecoder initialization with default parameters."""
        decoder = MLPDecoder(in_features=5, out_features=10)
        assert hasattr(decoder, "model")
        assert isinstance(decoder.model, nn.Sequential)

    def test_init_custom_hidden_dims(self):
        """Test MLPDecoder initialization with custom hidden dimensions."""
        hidden_dims = [16, 32, 64]
        decoder = MLPDecoder(in_features=5, out_features=10, hidden_dims=hidden_dims)

        # Count linear layers
        linear_layers = [layer for layer in decoder.model if isinstance(layer, nn.Linear)]
        assert len(linear_layers) == len(hidden_dims) + 1

    def test_init_custom_activation(self):
        """Test MLPDecoder initialization with custom activation."""
        custom_activation = nn.LeakyReLU()
        decoder = MLPDecoder(in_features=5, out_features=10, activation=custom_activation)

        # Check that LeakyReLU activations are used
        leaky_relu_layers = [layer for layer in decoder.model if isinstance(layer, nn.LeakyReLU)]
        assert len(leaky_relu_layers) > 0

    def test_init_output_activation(self):
        """Test MLPDecoder initialization with output activation."""
        output_activation = nn.Tanh()
        decoder = MLPDecoder(in_features=5, out_features=10, output_activation=output_activation)

        # Check that last layer is the output activation
        last_layer = list(decoder.model.children())[-1]
        assert isinstance(last_layer, nn.Tanh)

    def test_init_no_hidden_dims(self):
        """Test MLPDecoder initialization without hidden dimensions."""
        decoder = MLPDecoder(in_features=5, out_features=10, hidden_dims=[])

        # Should only have one linear layer
        linear_layers = [layer for layer in decoder.model if isinstance(layer, nn.Linear)]
        assert len(linear_layers) == 1
        assert linear_layers[0].in_features == 5
        assert linear_layers[0].out_features == 10

    def test_forward_basic(self):
        """Test forward pass with basic configuration."""
        decoder = MLPDecoder(in_features=5, out_features=10)
        x = torch.randn(32, 5)

        output = decoder(x)
        assert output.shape == (32, 10)

    def test_forward_custom_hidden(self):
        """Test forward pass with custom hidden dimensions."""
        decoder = MLPDecoder(in_features=8, out_features=20, hidden_dims=[16, 32, 64])
        x = torch.randn(16, 8)

        output = decoder(x)
        assert output.shape == (16, 20)

    def test_forward_single_sample(self):
        """Test forward pass with single sample."""
        decoder = MLPDecoder(in_features=3, out_features=5)
        x = torch.randn(1, 3)

        output = decoder(x)
        assert output.shape == (1, 5)

    def test_forward_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        decoder = MLPDecoder(in_features=5, out_features=10)

        for batch_size in [1, 8, 32, 128]:
            x = torch.randn(batch_size, 5)
            output = decoder(x)
            assert output.shape == (batch_size, 10)

    def test_forward_with_output_activation(self):
        """Test forward pass with output activation."""
        decoder = MLPDecoder(in_features=5, out_features=10, output_activation=nn.Tanh())
        x = torch.randn(32, 5)

        output = decoder(x)
        assert output.shape == (32, 10)
        # Output should be in [-1, 1] range due to tanh
        assert torch.all(output >= -1) and torch.all(output <= 1)

    def test_forward_with_kwargs(self):
        """Test forward pass with additional kwargs."""
        decoder = MLPDecoder(in_features=5, out_features=10)
        x = torch.randn(32, 5)

        # Should handle extra kwargs gracefully
        output = decoder(x, some_extra_arg=42)
        assert output.shape == (32, 10)

    def test_parameter_count(self):
        """Test that parameter count is reasonable."""
        decoder = MLPDecoder(in_features=5, out_features=10, hidden_dims=[15])

        total_params = sum(p.numel() for p in decoder.parameters())
        # Expected: (5*15 + 15) + (15*10 + 10) = 90 + 160 = 250
        expected_params = (5 * 15 + 15) + (15 * 10 + 10)
        assert total_params == expected_params

    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        decoder = MLPDecoder(in_features=5, out_features=10)
        x = torch.randn(32, 5, requires_grad=True)

        output = decoder(x)
        loss = output.sum()
        loss.backward()

        # Check that input gradients exist
        assert x.grad is not None
        # Check that model parameters have gradients
        for param in decoder.parameters():
            assert param.grad is not None


class TestMLPEncoderDecoder:
    """Test MLPEncoder and MLPDecoder working together."""

    def test_encoder_decoder_chain(self):
        """Test encoder-decoder chain."""
        encoder = MLPEncoder(in_features=20, out_features=5, hidden_dims=[10])
        decoder = MLPDecoder(in_features=5, out_features=20, hidden_dims=[10])

        x = torch.randn(32, 20)
        encoded = encoder(x)
        decoded = decoder(encoded)

        assert encoded.shape == (32, 5)
        assert decoded.shape == (32, 20)

    def test_encoder_decoder_reconstruction(self):
        """Test that encoder-decoder can learn to reconstruct."""
        # Simple test - not training, just checking dimensions work
        encoder = MLPEncoder(in_features=10, out_features=3)
        decoder = MLPDecoder(in_features=3, out_features=10)

        x = torch.randn(16, 10)

        # Forward pass
        encoded = encoder(x)
        reconstructed = decoder(encoded)

        assert reconstructed.shape == x.shape

        # Can compute loss
        loss = nn.MSELoss()(reconstructed, x)
        assert loss.item() >= 0

    def test_asymmetric_dimensions(self):
        """Test encoder-decoder with asymmetric dimensions."""
        encoder = MLPEncoder(in_features=50, out_features=8, hidden_dims=[32, 16])
        decoder = MLPDecoder(in_features=8, out_features=30, hidden_dims=[16, 32])

        x = torch.randn(64, 50)
        encoded = encoder(x)
        decoded = decoder(encoded)

        assert encoded.shape == (64, 8)
        assert decoded.shape == (64, 30)

    def test_bottleneck_architecture(self):
        """Test bottleneck architecture with very small latent dimension."""
        encoder = MLPEncoder(in_features=100, out_features=2, hidden_dims=[50, 20, 10])
        decoder = MLPDecoder(in_features=2, out_features=100, hidden_dims=[10, 20, 50])

        x = torch.randn(32, 100)
        encoded = encoder(x)
        reconstructed = decoder(encoded)

        assert encoded.shape == (32, 2)  # Very small bottleneck
        assert reconstructed.shape == (32, 100)

    def test_device_compatibility(self):
        """Test that models work on different devices."""
        encoder = MLPEncoder(in_features=10, out_features=5)
        decoder = MLPDecoder(in_features=5, out_features=10)

        # Test on CPU (default)
        x_cpu = torch.randn(16, 10)
        encoded_cpu = encoder(x_cpu)
        decoded_cpu = decoder(encoded_cpu)

        assert encoded_cpu.device.type == "cpu"
        assert decoded_cpu.device.type == "cpu"

    def test_different_dtypes(self):
        """Test that models work with different data types."""
        encoder = MLPEncoder(in_features=10, out_features=5)
        decoder = MLPDecoder(in_features=5, out_features=10)

        # Test with float32
        x_float32 = torch.randn(16, 10, dtype=torch.float32)
        encoded = encoder(x_float32)
        decoded = decoder(encoded)

        assert encoded.dtype == torch.float32
        assert decoded.dtype == torch.float32
