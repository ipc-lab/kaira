import pytest
import torch

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
    return DeepJSCCFeedbackModel(
        channel_snr=10,
        conv_depth=64,
        channel_type="awgn",
        feedback_snr=10,
        refinement_layer=False,
        layer_id=0,
    )


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
    input_tensor = torch.randn(4, 64, 8, 8)
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
