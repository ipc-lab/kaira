import pytest
import torch

from kaira.channels import AWGNChannel
from kaira.constraints import TotalPowerConstraint
from kaira.models.image.yilmaz2024_deepjscc_wz import (
    Yilmaz2024DeepJSCCWZConditionalDecoder,
    Yilmaz2024DeepJSCCWZConditionalEncoder,
    Yilmaz2024DeepJSCCWZDecoder,
    Yilmaz2024DeepJSCCWZEncoder,
    Yilmaz2024DeepJSCCWZModel,
    Yilmaz2024DeepJSCCWZSmallDecoder,
    Yilmaz2024DeepJSCCWZSmallEncoder,
)


@pytest.fixture
def small_encoder():
    return Yilmaz2024DeepJSCCWZSmallEncoder(N=64, M=128)


@pytest.fixture
def small_decoder(small_encoder):
    return Yilmaz2024DeepJSCCWZSmallDecoder(N=64, M=128, encoder=small_encoder)


@pytest.fixture
def encoder():
    return Yilmaz2024DeepJSCCWZEncoder(N=64, M=128)


@pytest.fixture
def decoder():
    return Yilmaz2024DeepJSCCWZDecoder(N=64, M=128)


@pytest.fixture
def conditional_encoder():
    return Yilmaz2024DeepJSCCWZConditionalEncoder(N=64, M=128)


@pytest.fixture
def conditional_decoder():
    return Yilmaz2024DeepJSCCWZConditionalDecoder(N=64, M=128)


@pytest.fixture
def channel():
    return AWGNChannel(snr_db=10)


@pytest.fixture
def constraint():
    return TotalPowerConstraint(total_power=1.0)


@pytest.fixture
def sample_image():
    return torch.randn(1, 3, 256, 256)


@pytest.fixture
def side_info():
    return torch.randn(1, 3, 256, 256)


def test_small_encoder_initialization(small_encoder):
    assert isinstance(small_encoder, Yilmaz2024DeepJSCCWZSmallEncoder)


def test_small_decoder_initialization(small_decoder):
    assert isinstance(small_decoder, Yilmaz2024DeepJSCCWZSmallDecoder)


def test_encoder_initialization(encoder):
    assert isinstance(encoder, Yilmaz2024DeepJSCCWZEncoder)


def test_decoder_initialization(decoder):
    assert isinstance(decoder, Yilmaz2024DeepJSCCWZDecoder)


def test_conditional_encoder_initialization(conditional_encoder):
    assert isinstance(conditional_encoder, Yilmaz2024DeepJSCCWZConditionalEncoder)


def test_conditional_decoder_initialization(conditional_decoder):
    assert isinstance(conditional_decoder, Yilmaz2024DeepJSCCWZConditionalDecoder)


def test_small_encoder_forward(small_encoder, sample_image):
    csi = torch.ones(1, 1, 1, 1)
    encoded_image = small_encoder(sample_image, csi)
    assert isinstance(encoded_image, torch.Tensor)
    assert encoded_image.shape == (1, 128, 16, 16)


def test_small_decoder_forward(small_decoder, small_encoder, sample_image, side_info):
    csi = torch.ones(1, 1, 1, 1)
    encoded_image = small_encoder(sample_image, csi)
    decoded_image = small_decoder(encoded_image, side_info, csi)
    assert isinstance(decoded_image, torch.Tensor)
    assert decoded_image.shape == sample_image.shape


def test_encoder_forward(encoder, sample_image):
    csi = torch.ones(1, 1, 1, 1)
    encoded_image = encoder(sample_image, csi)
    assert isinstance(encoded_image, torch.Tensor)
    assert encoded_image.shape == (1, 128, 16, 16)


def test_decoder_forward(decoder, encoder, sample_image, side_info):
    csi = torch.ones(1, 1, 1, 1)
    encoded_image = encoder(sample_image, csi)
    decoded_image = decoder(encoded_image, side_info, csi)
    assert isinstance(decoded_image, torch.Tensor)
    assert decoded_image.shape == sample_image.shape


def test_conditional_encoder_forward(conditional_encoder, sample_image, side_info):
    csi = torch.ones(1, 1, 1, 1)
    encoded_image = conditional_encoder(sample_image, side_info, csi)
    assert isinstance(encoded_image, torch.Tensor)
    assert encoded_image.shape == (1, 128, 16, 16)


def test_conditional_decoder_forward(conditional_decoder, conditional_encoder, sample_image, side_info):
    csi = torch.ones(1, 1, 1, 1)
    encoded_image = conditional_encoder(sample_image, side_info, csi)
    decoded_image = conditional_decoder(encoded_image, side_info, csi)
    assert isinstance(decoded_image, torch.Tensor)
    assert decoded_image.shape == sample_image.shape


def test_yilmaz2024_deepjscc_wz_model_initialization(encoder, channel, decoder, constraint):
    model = Yilmaz2024DeepJSCCWZModel(encoder=encoder, channel=channel, decoder=decoder, constraint=constraint)
    assert isinstance(model, Yilmaz2024DeepJSCCWZModel)


def test_yilmaz2024_deepjscc_wz_model_forward(encoder, channel, decoder, constraint, sample_image, side_info):
    model = Yilmaz2024DeepJSCCWZModel(encoder=encoder, channel=channel, decoder=decoder, constraint=constraint)
    csi = torch.ones(1, 1, 1, 1)
    # The model forward now returns the decoded tensor directly
    decoded_image = model(sample_image, side_info, csi=csi)
    # Check if the output is a tensor
    assert isinstance(decoded_image, torch.Tensor)
    # Check if the output shape matches the input image shape
    assert decoded_image.shape == sample_image.shape
