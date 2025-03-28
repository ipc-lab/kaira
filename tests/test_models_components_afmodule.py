import torch
import pytest
from kaira.models.components.afmodule import AFModule

@pytest.fixture
def feature_map():
    """Create a feature map tensor with the expected shape [batch, channels, height, width]."""
    batch_size = 4
    channels = 16  # This is the N parameter in AFModule
    height = 32
    width = 32
    torch.manual_seed(42)
    return torch.randn(batch_size, channels, height, width)

@pytest.fixture
def side_info():
    """Create side information tensor (e.g., channel state information)."""
    batch_size = 4
    csi_length = 1
    torch.manual_seed(43)
    return torch.randn(batch_size, csi_length)

def test_afmodule_initialization():
    """Test AFModule initialization."""
    N = 16
    csi_length = 1
    
    module = AFModule(N=N, csi_length=csi_length)
    
    # Check attributes
    assert module.c_in == N
    
    # Check module structure
    assert isinstance(module.layers, torch.nn.Sequential)
    assert len(module.layers) == 4  # 2 linear layers and 2 activation functions
    
    # Check first linear layer dimensions
    first_linear = module.layers[0]
    assert isinstance(first_linear, torch.nn.Linear)
    assert first_linear.in_features == N + csi_length
    assert first_linear.out_features == N

def test_afmodule_forward(feature_map, side_info):
    """Test AFModule forward pass."""
    batch_size, channels, height, width = feature_map.shape
    _, csi_length = side_info.shape
    
    # Initialize module
    module = AFModule(N=channels, csi_length=csi_length)
    
    # Create input tuple (feature_map, side_info)
    inputs = (feature_map, side_info)
    
    # Forward pass
    output = module(inputs)
    
    # Check output shape
    assert output.shape == feature_map.shape
    
    # Check output range (should be bounded due to sigmoid in the last layer multiplied by input)
    assert torch.all(output <= feature_map)  # Element-wise multiplication can only decrease absolute values
    
    # Confirm the mechanism is working by checking that output is different from input
    assert not torch.allclose(output, feature_map)

def test_afmodule_mask_creation(feature_map, side_info):
    """Test that the mask creation in AFModule behaves correctly."""
    batch_size, channels, height, width = feature_map.shape
    _, csi_length = side_info.shape
    
    # Initialize module
    module = AFModule(N=channels, csi_length=csi_length)
    
    # Forward pass with synthetic inputs
    inputs = (feature_map, side_info)
    
    # We'll manually compute the steps to verify the behavior
    # 1. Mean pooling across spatial dimensions
    context = torch.mean(feature_map, dim=(2, 3))
    assert context.shape == (batch_size, channels)
    
    # 2. Concatenate with side information
    context_input = torch.cat([context, side_info], dim=1)
    assert context_input.shape == (batch_size, channels + csi_length)
    
    # 3. Apply the layers to get the mask
    mask = module.layers(context_input).view(batch_size, channels, 1, 1)
    assert mask.shape == (batch_size, channels, 1, 1)
    
    # 4. Check that mask values are between 0 and 1 (due to sigmoid)
    assert torch.all(mask >= 0) and torch.all(mask <= 1)
    
    # Run the full module to compare
    output = module(inputs)
    expected_output = mask * feature_map
    
    # Check that output matches our manual computation
    assert torch.allclose(output, expected_output)

def test_afmodule_with_different_dimensions():
    """Test AFModule with different N and csi_length values."""
    test_configs = [
        {"N": 8, "csi_length": 1, "batch_size": 2, "height": 16, "width": 16},
        {"N": 32, "csi_length": 4, "batch_size": 1, "height": 8, "width": 8},
        {"N": 64, "csi_length": 8, "batch_size": 3, "height": 24, "width": 24}
    ]
    
    for config in test_configs:
        # Create module with this configuration
        module = AFModule(N=config["N"], csi_length=config["csi_length"])
        
        # Create input tensors
        feature_map = torch.randn(config["batch_size"], config["N"], 
                                 config["height"], config["width"])
        side_info = torch.randn(config["batch_size"], config["csi_length"])
        
        # Run forward pass
        output = module((feature_map, side_info))
        
        # Check output shape
        assert output.shape == feature_map.shape