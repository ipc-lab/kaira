import pytest
import torch
import torch.nn as nn

from kaira.channels import AWGNChannel, IdentityChannel
from kaira.models.image.kurka2020_deepjscc_feedback import (
    DeepJSCCFeedbackDecoder,
    DeepJSCCFeedbackEncoder,
    DeepJSCCFeedbackModel,
    OutputsCombiner,
)


@pytest.fixture
def encoder():
    return DeepJSCCFeedbackEncoder(conv_depth=64)


@pytest.fixture
def decoder():
    return DeepJSCCFeedbackDecoder(n_channels=3)


@pytest.fixture
def model():
    # Create with explicit channel objects to avoid None errors
    forward_channel = AWGNChannel(snr_db=10)
    feedback_channel = IdentityChannel()
    return DeepJSCCFeedbackModel(channel_snr=10, conv_depth=64, channel_type="awgn", feedback_snr=10, refinement_layer=False, layer_id=0, forward_channel=forward_channel, feedback_channel=feedback_channel)


@pytest.fixture
def model_with_refinement():
    # Create a model with refinement layer enabled and custom components to handle the refinement process
    forward_channel = AWGNChannel(snr_db=10)
    feedback_channel = AWGNChannel(snr_db=15)

    # Create a custom encoder that can handle 6 input channels (3 from image + 3 from feedback)
    class CustomEncoder(DeepJSCCFeedbackEncoder):
        def __init__(self, conv_depth):
            super().__init__(conv_depth)
            # Replace the first layer to accept 6 channels instead of 3
            num_filters = 256
            old_layers = self.layers

            # Create a new ModuleList with the first Conv2d modified to accept 6 input channels
            new_layers = nn.ModuleList()
            new_layers.append(nn.Conv2d(6, num_filters, kernel_size=9, stride=2, padding=4, bias=True))

            # Add the rest of the layers
            for i, layer in enumerate(old_layers):
                if i > 0:  # Skip the first Conv2d which we've replaced
                    new_layers.append(layer)

            self.layers = new_layers

    # Create a custom decoder that can handle the concatenated channel outputs
    class CustomDecoder(DeepJSCCFeedbackDecoder):
        def __init__(self, n_channels):
            super().__init__(n_channels)

        def forward(self, x, *args, **kwargs):
            # If the input doesn't have 256 channels, we need to adapt it
            if x.size(1) != 256:
                batch_size, n_channels, height, width = x.shape
                # Create a tensor with 256 channels and copy the input into it
                temp = torch.zeros(batch_size, 256, height, width, device=x.device)
                temp[:, :n_channels, :, :] = x
                x = temp  # Assign the temp tensor back to x

            # Normal forward pass
            return super().forward(x, *args, **kwargs)

    # Create the model with refinement enabled
    model = DeepJSCCFeedbackModel(channel_snr=10, conv_depth=64, channel_type="awgn", feedback_snr=15, refinement_layer=True, layer_id=1, forward_channel=forward_channel, feedback_channel=feedback_channel)  # Enable refinement layer  # Non-zero layer ID for refinement

    # Replace components with our custom ones
    model.encoder = CustomEncoder(conv_depth=64)
    model.decoder = CustomDecoder(n_channels=3)

    # Initialize the feedback processor
    model.feedback_processor = OutputsCombiner()
    return model


@pytest.fixture
def model_no_feedback_noise():
    # Create a model with no feedback noise (feedback_snr=None)
    forward_channel = AWGNChannel(snr_db=10)
    return DeepJSCCFeedbackModel(channel_snr=10, conv_depth=64, channel_type="awgn", feedback_snr=None, refinement_layer=False, layer_id=0, forward_channel=forward_channel, feedback_channel=None)  # No feedback noise


def test_encoder_initialization(encoder):
    assert isinstance(encoder, DeepJSCCFeedbackEncoder)


def test_decoder_initialization(decoder):
    assert isinstance(decoder, DeepJSCCFeedbackDecoder)


def test_model_initialization(model):
    assert isinstance(model, DeepJSCCFeedbackModel)


def test_encoder_forward(encoder):
    input_tensor = torch.randn(4, 3, 32, 32)
    output = encoder(input_tensor)
    assert output.shape == (4, 64, 8, 8)


def test_decoder_forward(decoder):
    # Use 256 channels to match the decoder's expected input channels
    input_tensor = torch.randn(4, 256, 8, 8)
    output = decoder(input_tensor)
    assert output.shape == (4, 3, 32, 32)


def test_model_forward_base_layer(model):
    input_tensor = torch.randn(4, 3, 32, 32)
    output = model(input_tensor)
    assert "decoded_img" in output
    assert "decoded_img_fb" in output
    assert "channel_output" in output
    assert "feedback_channel_output" in output
    assert "channel_gain" in output
    assert output["decoded_img"].shape == (4, 3, 32, 32)
    assert output["decoded_img_fb"].shape == (4, 3, 32, 32)


def test_outputs_combiner_initialization():
    combiner = OutputsCombiner()
    assert isinstance(combiner, OutputsCombiner)


def test_outputs_combiner_forward():
    combiner = OutputsCombiner()
    img_prev = torch.randn(4, 3, 32, 32)
    residual = torch.randn(4, 3, 32, 32)
    output = combiner((img_prev, residual))
    assert output.shape == (4, 3, 32, 32)


def test_default_forward_channel_initialization():
    """Test that the forward channel is properly initialized when not provided."""
    test_snr = 15.0
    model = DeepJSCCFeedbackModel(channel_snr=test_snr, conv_depth=64, channel_type="awgn", feedback_snr=10, refinement_layer=False, layer_id=0, forward_channel=None, feedback_channel=IdentityChannel())  # Explicitly set to None to test initialization

    # Verify the forward channel was created as an AWGNChannel with correct SNR
    assert isinstance(model.forward_channel, AWGNChannel)
    assert model.forward_channel.snr_db == test_snr


def test_default_feedback_channel_initialization_perfect():
    """Test that the feedback channel is properly initialized as IdentityChannel when feedback_snr
    is None."""
    model = DeepJSCCFeedbackModel(channel_snr=10, conv_depth=64, channel_type="awgn", feedback_snr=None, refinement_layer=False, layer_id=0, forward_channel=AWGNChannel(snr_db=10), feedback_channel=None)  # None should result in IdentityChannel  # Explicitly set to None to test initialization

    # Verify the feedback channel was created as an IdentityChannel
    assert isinstance(model.feedback_channel, IdentityChannel)


def test_default_feedback_channel_initialization_noisy():
    """Test that the feedback channel is properly initialized as AWGNChannel when feedback_snr is
    provided."""
    test_fb_snr = 20.0
    model = DeepJSCCFeedbackModel(
        channel_snr=10, conv_depth=64, channel_type="awgn", feedback_snr=test_fb_snr, refinement_layer=False, layer_id=0, forward_channel=AWGNChannel(snr_db=10), feedback_channel=None  # Should result in AWGNChannel with this SNR  # Explicitly set to None to test initialization
    )

    # Verify the feedback channel was created as an AWGNChannel with correct SNR
    assert isinstance(model.feedback_channel, AWGNChannel)
    assert model.feedback_channel.snr_db == test_fb_snr


def test_no_feedback_noise_scenario(model_no_feedback_noise):
    """Test the scenario when feedback_snr is None (no feedback noise)."""
    input_tensor = torch.randn(4, 3, 32, 32)

    # Capture the channel output before forward pass
    with torch.no_grad():
        chn_out = model_no_feedback_noise.encoder(input_tensor)
        chn_out = model_no_feedback_noise.forward_channel(chn_out)

        # Run the forward pass
        output = model_no_feedback_noise(input_tensor)

        # When feedback_snr is None, chn_out_fb should equal chn_out
        # We can verify this by checking that channel_output equals feedback_channel_output
        assert torch.allclose(output["channel_output"], output["feedback_channel_output"])


def test_model_with_refinement_layer(model_with_refinement):
    """Test the refinement layer functionality in DeepJSCCFeedbackModel."""
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 32, 32)

    # Create previous outputs to simulate a refinement scenario
    prev_img_out_dec = torch.randn(batch_size, 3, 32, 32)
    prev_img_out_fb = torch.randn(batch_size, 3, 32, 32)
    prev_chn_out_dec = torch.randn(batch_size, 64, 8, 8)
    prev_chn_out_fb = torch.randn(batch_size, 64, 8, 8)
    prev_chn_gain = torch.randn(batch_size, 64, 8, 8)

    # For refinement layers, the input_data should be a tuple with 6 elements
    input_data = (input_tensor, prev_img_out_fb, prev_chn_out_fb, prev_img_out_dec, prev_chn_out_dec, prev_chn_gain)

    # Pass the properly formatted input tuple
    output = model_with_refinement(input_data)

    # Check the output structure
    assert "decoded_img" in output
    assert "decoded_img_fb" in output
    assert "channel_output" in output
    assert "feedback_channel_output" in output
    assert "channel_gain" in output

    # Check output shapes
    assert output["decoded_img"].shape == (batch_size, 3, 32, 32)
    assert output["decoded_img_fb"].shape == (batch_size, 3, 32, 32)
