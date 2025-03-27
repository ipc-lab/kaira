import pytest
import torch

from kaira.models.image.yang2024_deepjcc_swin import (
    SwinJSCCConfig,
    Yang2024DeepJSCCSwinDecoder,
    Yang2024DeepJSCCSwinEncoder,
    create_swin_jscc_models,
)


@pytest.fixture
def config():
    return SwinJSCCConfig.from_preset("tiny")


@pytest.fixture
def encoder(config):
    return Yang2024DeepJSCCSwinEncoder(**config.get_encoder_kwargs(C=16))


@pytest.fixture
def decoder(config):
    return Yang2024DeepJSCCSwinDecoder(**config.get_decoder_kwargs(C=16))


@pytest.fixture
def sample_image():
    return torch.randn(1, 3, 224, 224)


def test_encoder_initialization(encoder):
    assert isinstance(encoder, Yang2024DeepJSCCSwinEncoder)
    assert len(encoder.layers) == 4


def test_decoder_initialization(decoder):
    assert isinstance(decoder, Yang2024DeepJSCCSwinDecoder)
    assert len(decoder.layers) == 4


def test_encoder_forward(encoder, sample_image):
    encoded_image = encoder(sample_image)
    assert isinstance(encoded_image, torch.Tensor)
    assert encoded_image.shape == (1, 16, 14, 14)


def test_decoder_forward(decoder, encoder, sample_image):
    encoded_image = encoder(sample_image)
    decoded_image = decoder(encoded_image)
    assert isinstance(decoded_image, torch.Tensor)
    assert decoded_image.shape == sample_image.shape


def test_encoder_decoder_roundtrip(encoder, decoder, sample_image):
    encoded_image = encoder(sample_image)
    decoded_image = decoder(encoded_image)
    assert not torch.equal(decoded_image, sample_image)
    assert decoded_image.shape == sample_image.shape
    assert torch.all(decoded_image > -5) and torch.all(decoded_image < 5)


@pytest.mark.parametrize("img_size", [(128, 128), (256, 256)])
def test_encoder_decoder_different_sizes(config, img_size):
    config.img_size = img_size
    encoder = Yang2024DeepJSCCSwinEncoder(**config.get_encoder_kwargs(C=16))
    decoder = Yang2024DeepJSCCSwinDecoder(**config.get_decoder_kwargs(C=16))
    sample_image = torch.randn(1, 3, *img_size)
    encoded_image = encoder(sample_image)
    decoded_image = decoder(encoded_image)
    assert decoded_image.shape == sample_image.shape


def test_swin_jscc_models_creation(config):
    encoder, decoder = create_swin_jscc_models(config, channel_dim=16)
    assert isinstance(encoder, Yang2024DeepJSCCSwinEncoder)
    assert isinstance(decoder, Yang2024DeepJSCCSwinDecoder)


def test_swin_jscc_models_forward(config, sample_image):
    encoder, decoder = create_swin_jscc_models(config, channel_dim=16)
    encoded_image = encoder(sample_image)
    decoded_image = decoder(encoded_image)
    assert decoded_image.shape == sample_image.shape
