import pytest
import torch

from kaira.models.image.yang2024_deepjcc_swin import (
    SwinJSCCConfig,
    Yang2024DeepJSCCSwinDecoder,
    Yang2024DeepJSCCSwinEncoder,
    create_swin_jscc_models,
)

# Add MockLogger to handle missing logger attribute
class MockLogger:
    def info(self, msg):
        pass
    def debug(self, msg):
        pass
    def warning(self, msg):
        pass
    def error(self, msg):
        pass

@pytest.fixture
def config():
    # Create a custom config rather than using preset to ensure lengths match
    # Note: embed_dims must have one more element than depths/num_heads
    return SwinJSCCConfig(
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dims=[96, 192, 384, 768, 1536],  # One more element than depths/num_heads
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
    )


@pytest.fixture
def encoder(config):
    model = Yang2024DeepJSCCSwinEncoder(**config.get_encoder_kwargs(C=16))
    # Add mock logger to handle missing attribute
    model.logger = MockLogger()
    return model


@pytest.fixture
def decoder(config):
    # Make sure to remove 'in_chans' from decoder kwargs if it exists
    decoder_kwargs = config.get_decoder_kwargs(C=16)
    if 'in_chans' in decoder_kwargs:
        decoder_kwargs.pop('in_chans')
    
    model = Yang2024DeepJSCCSwinDecoder(**decoder_kwargs)
    # Add mock logger to handle missing attribute
    model.logger = MockLogger()
    return model


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
    # The output shape depends on the model configuration
    # Check only batch size and channel dimensions
    assert encoded_image.shape[0] == 1
    assert encoded_image.shape[-3] == 16


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
def test_encoder_decoder_different_sizes(img_size):
    # Create a custom config with correct lengths for each test case
    config = SwinJSCCConfig(
        img_size=img_size,
        patch_size=4,
        in_chans=3,
        embed_dims=[96, 192, 384, 768, 1536],  # One more element than depths/num_heads
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
    )
    
    # Create encoder and decoder with mock loggers
    encoder = Yang2024DeepJSCCSwinEncoder(**config.get_encoder_kwargs(C=16))
    encoder.logger = MockLogger()
    
    decoder_kwargs = config.get_decoder_kwargs(C=16)
    if 'in_chans' in decoder_kwargs:
        decoder_kwargs.pop('in_chans')
    
    decoder = Yang2024DeepJSCCSwinDecoder(**decoder_kwargs)
    decoder.logger = MockLogger()
    
    sample_image = torch.randn(1, 3, *img_size)
    encoded_image = encoder(sample_image)
    decoded_image = decoder(encoded_image)
    assert decoded_image.shape == sample_image.shape


def test_swin_jscc_models_creation():
    config = SwinJSCCConfig(
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dims=[96, 192, 384, 768, 1536],  # One more element than depths/num_heads
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
    )
    
    # Create models
    encoder, decoder = create_swin_jscc_models(config, channel_dim=16)
    
    # Add mock loggers
    encoder.logger = MockLogger()
    decoder.logger = MockLogger()
    
    assert isinstance(encoder, Yang2024DeepJSCCSwinEncoder)
    assert isinstance(decoder, Yang2024DeepJSCCSwinDecoder)


def test_swin_jscc_models_forward():
    config = SwinJSCCConfig(
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dims=[96, 192, 384, 768, 1536],  # One more element than depths/num_heads
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
    )
    
    # Create models and add mock loggers
    encoder, decoder = create_swin_jscc_models(config, channel_dim=16)
    encoder.logger = MockLogger()
    decoder.logger = MockLogger()
    
    sample_image = torch.randn(1, 3, 224, 224)
    encoded_image = encoder(sample_image)
    decoded_image = decoder(encoded_image)
    assert decoded_image.shape == sample_image.shape
