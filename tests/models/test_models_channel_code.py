"""Tests for the ChannelCodeModel class."""
import pytest
import torch
import numpy as np
from kaira.models import ChannelCodeModel
from kaira.channels import IdentityChannel
from kaira.constraints import IdentityConstraint
from kaira.modulations import IdentityModulator, IdentityDemodulator
from kaira.models.generic import IdentityModel


class TestChannelCodeModel:
    """Test suite for the ChannelCodeModel class."""

    @pytest.fixture
    def channel_code_model(self):
        """Create a simple channel code model for testing."""
        # Create simple components for the channel code model
        encoder = IdentityModel()
        decoder = IdentityModel()
        constraint = IdentityConstraint()
        modulator = IdentityModulator()
        channel = IdentityChannel()
        demodulator = IdentityDemodulator()

        # Create the channel code model
        model = ChannelCodeModel(
            encoder=encoder,
            constraint=constraint,
            modulator=modulator,
            channel=channel,
            demodulator=demodulator,
            decoder=decoder,
        )
        return model

    def test_channel_code_model_init(self, channel_code_model):
        """Test the initialization of the ChannelCodeModel."""
        # Verify that the model has the correct components
        assert isinstance(channel_code_model.encoder, IdentityModel)
        assert isinstance(channel_code_model.decoder, IdentityModel)
        assert isinstance(channel_code_model.constraint, IdentityConstraint)
        assert isinstance(channel_code_model.modulator, IdentityModulator)
        assert isinstance(channel_code_model.channel, IdentityChannel)
        assert isinstance(channel_code_model.demodulator, IdentityDemodulator)

        # Verify that the steps are correctly set
        assert len(channel_code_model.steps) == 6
        assert channel_code_model.steps[0] == channel_code_model.encoder
        assert channel_code_model.steps[1] == channel_code_model.modulator
        assert channel_code_model.steps[2] == channel_code_model.constraint
        assert channel_code_model.steps[3] == channel_code_model.channel
        assert channel_code_model.steps[4] == channel_code_model.demodulator
        assert channel_code_model.steps[5] == channel_code_model.decoder

    def test_channel_code_model_forward(self, channel_code_model):
        """Test the forward method of the ChannelCodeModel."""
        # Create a test input tensor
        batch_size = 2
        input_dim = 10
        input_data = torch.rand(batch_size, input_dim)
        
        # Process the input through the model
        output = channel_code_model(input_data)
        
        # Check the output type and structure
        assert isinstance(output, dict)
        assert "final_output" in output
        assert "history" in output
        
        # Check the output values
        assert torch.allclose(output["final_output"], input_data)
        assert len(output["history"]) == 1
        
        # Check the history entries
        history_entry = output["history"][0]
        assert "encoded" in history_entry
        assert "received" in history_entry
        assert "decoded" in history_entry
        assert "soft_estimate" in history_entry
        
        # With identity components, input should pass through unchanged
        assert torch.allclose(history_entry["encoded"], input_data)
        assert torch.allclose(history_entry["received"], input_data)
        assert torch.allclose(history_entry["decoded"], input_data)

    def test_channel_code_model_end_to_end(self):
        """Test a more realistic end-to-end scenario with simple implementations."""
        # Define a simple encoder/decoder that adds/removes a parity bit
        class SimpleEncoder(torch.nn.Module):
            def forward(self, x, *args, **kwargs):
                # Add a parity bit (sum of all elements % 2)
                parity = (torch.sum(x, dim=1) % 2).unsqueeze(1)
                return torch.cat([x, parity], dim=1)
        
        class SimpleDecoder(torch.nn.Module):
            def forward(self, x, *args, **kwargs):
                # Remove the parity bit and return the original data
                original = x[:, :-1]
                # In a real decoder, we would check the parity bit here
                soft_estimate = original  # For simplicity
                return original, soft_estimate
        
        # Create model components
        encoder = SimpleEncoder()
        decoder = SimpleDecoder()
        constraint = IdentityConstraint()
        modulator = IdentityModulator()
        channel = IdentityChannel()
        demodulator = IdentityDemodulator()
        
        # Create the channel code model
        model = ChannelCodeModel(
            encoder=encoder,
            constraint=constraint,
            modulator=modulator,
            channel=channel,
            demodulator=demodulator,
            decoder=decoder,
        )
        
        # Create a test input tensor
        batch_size = 3
        input_dim = 5
        input_data = torch.rand(batch_size, input_dim)
        
        # Process the input through the model
        output = model(input_data)
        
        # Check the output
        assert torch.allclose(output["final_output"], input_data)
        assert len(output["history"]) == 1
        
        # Check the encoded data has the extra parity bit
        encoded = output["history"][0]["encoded"]
        assert encoded.shape == (batch_size, input_dim + 1)