import pytest
import torch

from kaira.models.base import BaseModel, CSIFormat, ChannelAwareBaseModel


# Helper classes for testing
class SimpleTestModel(ChannelAwareBaseModel):
    """A simple model for testing the ChannelAwareBaseModel functionality."""
    
    def __init__(self, expected_format=CSIFormat.LINEAR):
        super().__init__(expected_csi_format=expected_format)
    
    def forward(self, x, csi, *args, **kwargs):
        """Simple forward pass that multiplies input by CSI."""
        return x * csi


class NestedTestModel(ChannelAwareBaseModel):
    """A model that contains another channel-aware model as a component."""
    
    def __init__(self):
        super().__init__()
        self.submodel = SimpleTestModel(expected_format=CSIFormat.DB)
    
    def forward(self, x, csi, *args, **kwargs):
        """Forward pass that formats CSI and passes it to submodel."""
        formatted_csi = self.format_csi_for_submodules(csi, self.submodel)
        return self.submodel(x, formatted_csi)


# Test initialization and basic properties
def test_channel_aware_base_model_init():
    """Test initializing the ChannelAwareBaseModel with various parameters."""
    # Test with default parameters
    model = SimpleTestModel()
    assert model.expected_csi_format == CSIFormat.LINEAR
    assert model.expected_csi_dims is None
    assert model.csi_min_value == float('-inf')
    assert model.csi_max_value == float('inf')
    
    # Test with custom parameters
    model = SimpleTestModel(expected_format=CSIFormat.DB)
    assert model.expected_csi_format == CSIFormat.DB
    
    # Test with all parameters specified
    model = ChannelAwareBaseModel(
        expected_csi_dims=(1, 2),
        expected_csi_format=CSIFormat.NORMALIZED,
        csi_min_value=0.0,
        csi_max_value=1.0
    )
    assert model.expected_csi_dims == (1, 2)
    assert model.expected_csi_format == CSIFormat.NORMALIZED
    assert model.csi_min_value == 0.0
    assert model.csi_max_value == 1.0


# Test validation functionality
def test_validate_csi():
    """Test the CSI validation functionality."""
    model = ChannelAwareBaseModel(
        expected_csi_dims=(1,),
        expected_csi_format=CSIFormat.LINEAR,
        csi_min_value=0.0,
        csi_max_value=10.0
    )
    
    # Valid CSI
    valid_csi = torch.tensor([[5.0]])
    assert model.validate_csi(valid_csi)
    
    # Invalid dimension
    invalid_dim_csi = torch.tensor([5.0, 6.0])
    assert not model.validate_csi(invalid_dim_csi)
    
    # Invalid value range
    invalid_range_csi = torch.tensor([[-1.0]])
    assert not model.validate_csi(invalid_range_csi)
    
    # Test with NaN - should raise ValueError
    nan_csi = torch.tensor([[float('nan')]])
    with pytest.raises(ValueError):
        model.validate_csi(nan_csi)
    
    # Test with wrong type - should raise ValueError
    with pytest.raises(ValueError):
        model.validate_csi(5.0)  # Not a tensor


# Test normalization functionality
def test_normalize_csi():
    """Test the CSI normalization functionality."""
    # Model expecting linear format
    linear_model = ChannelAwareBaseModel(expected_csi_format=CSIFormat.LINEAR)
    
    # Test converting from dB to linear
    db_csi = torch.tensor([0.0, 10.0, 20.0])
    normalized_csi = linear_model.normalize_csi(db_csi)
    expected_linear = torch.tensor([1.0, 10.0, 100.0])
    assert torch.allclose(normalized_csi, expected_linear)
    
    # Model expecting dB format
    db_model = ChannelAwareBaseModel(expected_csi_format=CSIFormat.DB)
    
    # Test converting from linear to dB
    linear_csi = torch.tensor([1.0, 10.0, 100.0])
    normalized_csi = db_model.normalize_csi(linear_csi)
    expected_db = torch.tensor([0.0, 10.0, 20.0])
    assert torch.allclose(normalized_csi, expected_db, atol=1e-5)
    
    # Model expecting normalized format
    norm_model = ChannelAwareBaseModel(expected_csi_format=CSIFormat.NORMALIZED)
    
    # Test normalizing to [0, 1] range
    unnorm_csi = torch.tensor([-10.0, 0.0, 10.0])
    normalized_csi = norm_model.normalize_csi(unnorm_csi)
    expected_norm = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(normalized_csi, expected_norm)


# Test conversion utility functions
def test_conversion_functions():
    """Test the CSI conversion utility functions."""
    model = ChannelAwareBaseModel()
    
    # Test dB to linear conversion
    db_values = torch.tensor([-10.0, 0.0, 10.0, 20.0])
    linear_values = model._db_to_linear(db_values)
    expected_linear = torch.tensor([0.1, 1.0, 10.0, 100.0])
    assert torch.allclose(linear_values, expected_linear)
    
    # Test linear to dB conversion
    linear_values = torch.tensor([0.1, 1.0, 10.0, 100.0])
    db_values = model._linear_to_db(linear_values)
    expected_db = torch.tensor([-10.0, 0.0, 10.0, 20.0])
    assert torch.allclose(db_values, expected_db, atol=1e-5)
    
    # Test normalize to range
    values = torch.tensor([-10.0, 0.0, 10.0])
    normalized = model._normalize_to_range(values, 0.0, 1.0)
    expected = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(normalized, expected)
    
    # Test normalize single value
    single_value = torch.tensor([5.0])
    normalized = model._normalize_to_range(single_value, 0.0, 1.0)
    expected = torch.tensor([0.0])  # When min == max, returns target_min
    assert torch.allclose(normalized, expected)


# Test the format_csi_for_submodules function
def test_format_csi_for_submodules():
    """Test formatting CSI for different submodules."""
    parent_model = ChannelAwareBaseModel(expected_csi_format=CSIFormat.LINEAR)
    
    # Format for a channel-aware submodule expecting dB
    db_submodel = SimpleTestModel(expected_format=CSIFormat.DB)
    linear_csi = torch.tensor([1.0, 10.0, 100.0])
    formatted_csi = parent_model.format_csi_for_submodules(linear_csi, db_submodel)
    expected_db = torch.tensor([0.0, 10.0, 20.0])
    assert torch.allclose(formatted_csi, expected_db, atol=1e-5)
    
    # Format for a channel-aware submodule expecting normalized
    norm_submodel = SimpleTestModel(expected_format=CSIFormat.NORMALIZED)
    linear_csi = torch.tensor([0.1, 1.0, 10.0])
    formatted_csi = parent_model.format_csi_for_submodules(linear_csi, norm_submodel)
    expected_norm = torch.tensor([0.0, 0.1, 1.0])
    assert torch.allclose(formatted_csi, expected_norm, atol=1e-5)
    
    # Format for a non-channel-aware submodule
    regular_submodule = torch.nn.Linear(10, 10)
    linear_csi = torch.tensor([1.0, 10.0, 100.0])
    formatted_csi = parent_model.format_csi_for_submodules(linear_csi, regular_submodule)
    # Should return the CSI unchanged
    assert torch.allclose(formatted_csi, linear_csi)


# Test forward_csi_to_sequential
def test_forward_csi_to_sequential():
    """Test forwarding CSI through a sequence of modules."""
    model = ChannelAwareBaseModel()
    
    # Create a mix of channel-aware and regular modules
    ca_module1 = SimpleTestModel(expected_format=CSIFormat.LINEAR)
    regular_module = torch.nn.Linear(2, 2)
    ca_module2 = SimpleTestModel(expected_format=CSIFormat.DB)
    
    # Initialize the modules properly
    regular_module.weight.data = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    regular_module.bias.data = torch.zeros(2)
    
    modules = [ca_module1, regular_module, ca_module2]
    
    # Input and CSI
    x = torch.tensor([[1.0, 2.0]])
    csi = torch.tensor([[10.0]])  # Linear scale
    
    # Forward through modules
    result = model.forward_csi_to_sequential(x, modules, csi)
    
    # Expected result:
    # 1. ca_module1: x * csi = [1, 2] * 10 = [10, 20]
    # 2. regular_module: [10, 20] * [[2, 0], [0, 2]] = [20, 40]
    # 3. ca_module2: [20, 40] * 10 = [200, 400]
    expected = torch.tensor([[200.0, 400.0]])
    
    assert torch.allclose(result, expected)


# Test extract_csi_from_channel_output
def test_extract_csi_from_channel_output():
    """Test extracting CSI from channel output dictionaries."""
    model = ChannelAwareBaseModel()
    
    # Test extracting from 'csi' key
    output1 = {'signal': torch.tensor([1.0]), 'csi': torch.tensor([10.0])}
    csi1 = model.extract_csi_from_channel_output(output1)
    assert torch.allclose(csi1, torch.tensor([10.0]))
    
    # Test extracting from 'snr' key
    output2 = {'signal': torch.tensor([1.0]), 'snr': torch.tensor([20.0])}
    csi2 = model.extract_csi_from_channel_output(output2)
    assert torch.allclose(csi2, torch.tensor([20.0]))
    
    # Test extracting from 'h' key (channel coefficients)
    output3 = {'signal': torch.tensor([1.0]), 'h': torch.tensor([0.5])}
    csi3 = model.extract_csi_from_channel_output(output3)
    assert torch.allclose(csi3, torch.tensor([0.5]))
    
    # Test extracting when no CSI key is present
    output4 = {'signal': torch.tensor([1.0])}
    with pytest.raises(ValueError):
        model.extract_csi_from_channel_output(output4)


# Test nested models with different CSI format expectations
def test_nested_channel_aware_models():
    """Test nested channel-aware models with different CSI format expectations."""
    # Create outer model expecting linear CSI and inner model expecting dB CSI
    model = NestedTestModel()
    
    # Input and linear CSI
    x = torch.tensor([1.0, 2.0])
    linear_csi = torch.tensor([10.0])  # Linear scale
    
    # Process data
    result = model(x, linear_csi)
    
    # Expected result:
    # 1. Convert linear CSI to dB: 10 dB
    # 2. Apply to input: [1, 2] * 10 = [10, 20]
    expected = torch.tensor([10.0, 20.0])
    
    assert torch.allclose(result, expected)