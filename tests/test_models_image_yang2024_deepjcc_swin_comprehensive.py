import pytest
import torch
import torch.nn as nn
from kaira.models.image.yang2024_deepjcc_swin import (
    Yang2024DeepJSCCSwinEncoder,
    Yang2024DeepJSCCSwinDecoder,
    SwinJSCCConfig,
    create_swin_jscc_models,
    _Mlp,
    _WindowAttention,
    _PatchEmbed,
    _PatchMerging,
    _SwinTransformerBlock,
    _BasicLayer,
    _AdaptiveModulator,
    _window_partition,
    _window_reverse
)

@pytest.fixture
def device():
    """Device fixture for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def small_config():
    """Fixture for a small model configuration for fast testing."""
    return {
        "img_size": 32,
        "patch_size": 2,
        "in_chans": 3,
        "embed_dims": [32, 64, 128],
        "depths": [2, 2],
        "num_heads": [2, 4],
        "window_size": 4,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_scale": None,
        "use_mixed_precision": False,
        "memory_efficient": False,
        "adaptation_hidden_factor": 1.5,
        "adaptation_layers": 2
    }

@pytest.fixture
def tiny_encoder(small_config, device):
    """Fixture for a tiny encoder model."""
    model = Yang2024DeepJSCCSwinEncoder(
        **small_config,
        C=16,
        bottleneck_dim=8,
        patch_norm=True
    ).to(device)
    return model

@pytest.fixture
def tiny_decoder(small_config, device):
    """Fixture for a tiny decoder model."""
    model = Yang2024DeepJSCCSwinDecoder(
        **small_config,
        C=16,
        bottleneck_dim=8,
        patch_norm=True,
        ape=False
    ).to(device)
    return model

@pytest.fixture
def sample_image(device):
    """Fixture providing a sample image for testing."""
    return torch.randn(2, 3, 32, 32).to(device)

# Test helper components first
def test_mlp_component():
    """Test the MLP component works correctly."""
    mlp = _Mlp(in_features=64, hidden_features=256, out_features=64)
    x = torch.randn(2, 10, 64)  # [B, N, C]
    
    # Test forward pass
    out = mlp(x)
    assert out.shape == x.shape
    
    # Test with dropout
    mlp_with_dropout = _Mlp(in_features=64, hidden_features=256, out_features=64, drop=0.1)
    out_with_dropout = mlp_with_dropout(x)
    assert out_with_dropout.shape == x.shape

def test_window_partition_and_reverse():
    """Test window partitioning and reversal functions."""
    # Create test input
    B, H, W, C = 2, 16, 16, 32
    x = torch.randn(B, H, W, C)
    window_size = 4
    
    # Test partitioning
    windows = _window_partition(x, window_size)
    num_windows = (H // window_size) * (W // window_size) * B
    assert windows.shape == (num_windows, window_size, window_size, C)
    
    # Test reversal
    x_reversed = _window_reverse(windows, window_size, H, W)
    assert x_reversed.shape == x.shape
    assert torch.allclose(x, x_reversed, rtol=1e-5)

def test_patch_embed():
    """Test the patch embedding module."""
    # Create module
    embed = _PatchEmbed(
        img_size=32,
        patch_size=2,
        in_chans=3,
        embed_dim=96
    )
    
    # Test parameters
    assert embed.patches_resolution == [16, 16]
    assert embed.num_patches == 256
    
    # Test forward
    x = torch.randn(2, 3, 32, 32)
    out = embed(x)
    assert out.shape == (2, 256, 96)
    
    # Test with norm
    embed_with_norm = _PatchEmbed(
        img_size=32,
        patch_size=2,
        in_chans=3,
        embed_dim=96,
        norm_layer=nn.LayerNorm
    )
    out_with_norm = embed_with_norm(x)
    assert out_with_norm.shape == (2, 256, 96)
    
    # Test FLOPs calculation
    flops = embed.flops()
    assert flops > 0

def test_patch_merging():
    """Test the patch merging module."""
    merge = _PatchMerging(
        input_resolution=(16, 16),
        dim=96
    )
    
    # Test forward
    x = torch.randn(2, 256, 96)  # [B, H*W, C]
    out = merge(x)
    assert out.shape == (2, 64, 96*2)  # Output has H*W/4 tokens, 2*C channels
    
    # Test FLOPs calculation
    flops = merge.flops()
    assert flops > 0

def test_window_attention():
    """Test the window attention module."""
    attn = _WindowAttention(
        dim=96,
        window_size=(4, 4),
        num_heads=3
    )
    
    # Test forward without mask
    x = torch.randn(8, 16, 96)  # [num_windows*B, N, C]
    out = attn(x)
    assert out.shape == x.shape
    
    # Test forward with mask
    mask = torch.zeros(1, 16, 16)
    out_with_mask = attn(x, mask=mask)
    assert out_with_mask.shape == x.shape
    
    # Test with token position bias
    out_with_token = attn(x, add_token=True, token_num=2)
    assert out_with_token.shape == x.shape
    
    # Test FLOPs calculation
    flops = attn.flops(16)
    assert flops > 0

def test_swin_transformer_block():
    """Test the Swin Transformer block."""
    block = _SwinTransformerBlock(
        dim=96,
        input_resolution=(16, 16),
        num_heads=3,
        window_size=4,
        shift_size=0
    )
    
    # Test forward
    x = torch.randn(2, 256, 96)  # [B, H*W, C]
    out = block(x)
    assert out.shape == x.shape
    
    # Test with shifted windows
    block_shifted = _SwinTransformerBlock(
        dim=96,
        input_resolution=(16, 16),
        num_heads=3,
        window_size=4,
        shift_size=2
    )
    out_shifted = block_shifted(x)
    assert out_shifted.shape == x.shape
    
    # Test FLOPs calculation
    flops = block.flops()
    assert flops > 0

def test_basic_layer():
    """Test the basic Swin Transformer layer."""
    layer = _BasicLayer(
        dim=96,
        out_dim=192,
        input_resolution=(32, 32),
        depth=2,
        num_heads=3,
        window_size=4,
        downsample=_PatchMerging
    )
    
    # Test forward
    x = torch.randn(2, 1024, 96)  # [B, H*W, C]
    out = layer(x)
    assert out.shape[0] == 2
    assert out.shape[1] == 256  # H*W/4
    assert out.shape[2] == 192  # out_dim
    
    # Test FLOPs calculation
    flops = layer.flops()
    assert flops > 0

def test_adaptive_modulator():
    """Test the adaptive modulator for SNR/rate adaptation."""
    mod = _AdaptiveModulator(hidden_dim=32)
    
    # Test forward
    batch_size = 4
    x = torch.randn(batch_size, 1)  # [B, 1] - SNR or rate values
    out = mod(x)
    assert out.shape == (batch_size, 32)
    assert torch.all((out >= 0) & (out <= 1))  # Output should be in [0,1] due to sigmoid

# Now test the main model components
def test_encoder_initialization(small_config):
    """Test encoder initialization with various parameters."""
    # Basic initialization
    encoder = Yang2024DeepJSCCSwinEncoder(
        **small_config,
        C=16,
        bottleneck_dim=8,
        patch_norm=True
    )
    assert isinstance(encoder, Yang2024DeepJSCCSwinEncoder)
    
    # Test with different window sizes
    encoder_small_window = Yang2024DeepJSCCSwinEncoder(
        **small_config,
        C=16,
        window_size=2
    )
    assert encoder_small_window.window_size == 2
    
    # Test model size calculation
    size_info = encoder.get_model_size()
    assert "total_params" in size_info
    assert "trainable_params" in size_info
    assert "param_size_mb" in size_info
    assert "flops_g" in size_info

def test_decoder_initialization(small_config):
    """Test decoder initialization with various parameters."""
    # Basic initialization
    decoder = Yang2024DeepJSCCSwinDecoder(
        **small_config,
        C=16,
        bottleneck_dim=8,
        patch_norm=True
    )
    assert isinstance(decoder, Yang2024DeepJSCCSwinDecoder)
    
    # Test with absolute position embedding
    decoder_with_ape = Yang2024DeepJSCCSwinDecoder(
        **small_config,
        C=16,
        ape=True
    )
    assert decoder_with_ape.ape == True
    assert hasattr(decoder_with_ape, "absolute_pos_embed")
    
    # Test model size calculation
    size_info = decoder.get_model_size()
    assert "total_params" in size_info
    assert "trainable_params" in size_info
    assert "param_size_mb" in size_info
    assert "flops_g" in size_info

def test_encoder_forward(tiny_encoder, sample_image, device):
    """Test encoder forward pass with various modes."""
    # Basic forward
    out = tiny_encoder(sample_image)
    assert out.shape[0] == sample_image.shape[0]
    
    # Test with SNR adaptation
    out_snr = tiny_encoder(sample_image, snr=10.0, model_mode="SwinJSCC_w/_SA")
    assert out_snr.shape[0] == sample_image.shape[0]
    
    # Test with rate adaptation
    out_rate, mask = tiny_encoder(sample_image, rate=8, model_mode="SwinJSCC_w/_RA")
    assert out_rate.shape[0] == sample_image.shape[0]
    assert mask.shape[0] == sample_image.shape[0]
    
    # Test with both adaptations
    out_both, mask_both = tiny_encoder(
        sample_image, 
        snr=10.0, 
        rate=8, 
        model_mode="SwinJSCC_w/_SAandRA"
    )
    assert out_both.shape[0] == sample_image.shape[0]
    assert mask_both.shape[0] == sample_image.shape[0]
    
    # Test with intermediate features
    out_with_features, features = tiny_encoder(
        sample_image,
        return_intermediate_features=True
    )
    assert isinstance(features, dict)
    assert "patch_embed" in features
    assert "norm" in features
    
    # Test updating resolution
    tiny_encoder.update_resolution(16, 16)
    out_updated = tiny_encoder(sample_image)
    assert out_updated.shape[0] == sample_image.shape[0]

def test_decoder_forward(tiny_decoder, device):
    """Test decoder forward pass with various modes."""
    # Create encoder output
    batch_size = 2
    encoder_output = torch.randn(batch_size, 64, 16).to(device)  # [B, H*W/16, C]
    
    # Basic forward
    out = tiny_decoder(encoder_output)
    assert out.shape == (batch_size, 3, 32, 32)
    
    # Test with SNR adaptation
    out_snr = tiny_decoder(encoder_output, snr=10.0, model_mode="SwinJSCC_w/_SA")
    assert out_snr.shape == (batch_size, 3, 32, 32)
    
    # Test with intermediate features
    out_with_features, features = tiny_decoder(
        encoder_output,
        return_intermediate_features=True
    )
    assert isinstance(features, dict)
    assert out_with_features.shape == (batch_size, 3, 32, 32)
    
    # Test updating resolution
    tiny_decoder.update_resolution(8, 8)
    out_updated = tiny_decoder(encoder_output)
    assert out_updated.shape[0] == batch_size
    assert out_updated.shape[1] == 3

def test_swin_jscc_config():
    """Test SwinJSCC configuration."""
    # Test basic configuration
    config = SwinJSCCConfig(
        img_size=32,
        patch_size=2,
        in_chans=3,
        embed_dims=[32, 64, 128],
        depths=[2, 2],
        num_heads=[2, 4]
    )
    
    # Test encoder kwargs
    encoder_kwargs = config.get_encoder_kwargs(C=16)
    assert encoder_kwargs["img_size"] == 32
    assert encoder_kwargs["C"] == 16
    
    # Test decoder kwargs
    decoder_kwargs = config.get_decoder_kwargs(C=16)
    assert decoder_kwargs["img_size"] == 32
    assert decoder_kwargs["C"] == 16
    
    # Test preset configurations
    tiny_config = SwinJSCCConfig.from_preset("tiny")
    assert len(tiny_config.embed_dims) == 4
    assert tiny_config.window_size == 7
    
    small_config = SwinJSCCConfig.from_preset("small")
    assert small_config.depths[2] == 18
    
    # Test invalid preset
    with pytest.raises(ValueError):
        SwinJSCCConfig.from_preset("nonexistent")

def test_create_swin_jscc_models(device):
    """Test model creation helper function."""
    config = SwinJSCCConfig(
        img_size=32,
        patch_size=2,
        in_chans=3,
        embed_dims=[32, 64, 128],
        depths=[2, 2],
        num_heads=[2, 4]
    )
    
    encoder, decoder = create_swin_jscc_models(config, channel_dim=16, device=device)
    
    assert isinstance(encoder, Yang2024DeepJSCCSwinEncoder)
    assert isinstance(decoder, Yang2024DeepJSCCSwinDecoder)
    
    # Test a complete forward pass
    x = torch.randn(2, 3, 32, 32).to(device)
    encoded = encoder(x)
    decoded = decoder(encoded)
    
    assert decoded.shape == x.shape

def test_gradient_checkpointing(tiny_encoder, tiny_decoder):
    """Test gradient checkpointing functionality."""
    # Enable gradient checkpointing
    tiny_encoder.set_gradient_checkpointing(True)
    tiny_decoder.set_gradient_checkpointing(True)
    
    # Disable gradient checkpointing
    tiny_encoder.set_gradient_checkpointing(False)
    tiny_decoder.set_gradient_checkpointing(False)

def test_no_weight_decay_methods(tiny_encoder, tiny_decoder):
    """Test methods that identify parameters for weight decay exclusion."""
    # Test encoder methods
    no_decay = tiny_encoder.no_weight_decay()
    assert "absolute_pos_embed" in no_decay
    
    no_decay_kw = tiny_encoder.no_weight_decay_keywords()
    assert "relative_position_bias_table" in no_decay_kw
    
    # Test decoder methods
    no_decay = tiny_decoder.no_weight_decay()
    assert "absolute_pos_embed" in no_decay
    
    no_decay_kw = tiny_decoder.no_weight_decay_keywords()
    assert "relative_position_bias_table" in no_decay_kw

def test_end_to_end(tiny_encoder, tiny_decoder, sample_image):
    """Test end-to-end model pipeline with different modes."""
    # Base mode
    encoded = tiny_encoder(sample_image)
    decoded = tiny_decoder(encoded)
    assert decoded.shape == sample_image.shape
    
    # SNR adaptation mode
    encoded_snr = tiny_encoder(sample_image, snr=10.0, model_mode="SwinJSCC_w/_SA")
    decoded_snr = tiny_decoder(encoded_snr, snr=10.0, model_mode="SwinJSCC_w/_SA")
    assert decoded_snr.shape == sample_image.shape
    
    # Rate adaptation mode
    encoded_rate, _ = tiny_encoder(sample_image, rate=8, model_mode="SwinJSCC_w/_RA")
    decoded_rate = tiny_decoder(encoded_rate, model_mode="SwinJSCC_w/_RA")
    assert decoded_rate.shape == sample_image.shape
    
    # Joint adaptation mode
    encoded_joint, _ = tiny_encoder(
        sample_image, 
        snr=10.0, 
        rate=8, 
        model_mode="SwinJSCC_w/_SAandRA"
    )
    decoded_joint = tiny_decoder(
        encoded_joint,
        snr=10.0,
        model_mode="SwinJSCC_w/_SAandRA"
    )
    assert decoded_joint.shape == sample_image.shape