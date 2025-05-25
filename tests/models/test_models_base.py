import pytest
import torch

from kaira.models.base import BaseModel, ChannelAwareBaseModel, ConfigurableModel


class DummyModel(BaseModel):
    def forward(self, x):
        return x * 2


def test_base_model_forward():
    model = DummyModel()
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    output = model(input_tensor)
    assert torch.allclose(output, input_tensor * 2)


def test_configurable_model_add_step():
    model = ConfigurableModel()

    def step(x):
        return x + 1

    model.add_step(step)
    assert len(model.steps) == 1
    assert model.steps[0] == step


def test_configurable_model_remove_step():
    model = ConfigurableModel()

    def step1(x):
        return x + 1

    def step2(x):
        return x * 2

    model.add_step(step1).add_step(step2)
    model.remove_step(0)
    assert len(model.steps) == 1
    assert model.steps[0] == step2


def test_configurable_model_forward():
    model = ConfigurableModel()

    def step1(x):
        return x + 1

    def step2(x):
        return x * 2

    model.add_step(step1).add_step(step2)
    input_tensor = torch.tensor([1.0, 2.0, 3.0])
    output = model(input_tensor)
    expected_output = (input_tensor + 1) * 2
    assert torch.allclose(output, expected_output)


def test_configurable_model_remove_step_out_of_range():
    model = ConfigurableModel()
    with pytest.raises(IndexError):
        model.remove_step(0)


def test_base_model_abstract_forward():
    """Test that BaseModel's forward method raises NotImplementedError if not implemented in
    subclass."""

    # Create a class that inherits from BaseModel and implements forward
    # but calls the parent's implementation which should raise NotImplementedError
    class IncompleteModel(BaseModel):
        def forward(self, *args, **kwargs):
            # Call parent's forward method which should raise NotImplementedError
            return super().forward(*args, **kwargs)

    # Instantiate the incomplete model
    model = IncompleteModel()

    # Verify that calling forward raises NotImplementedError
    with pytest.raises(NotImplementedError):
        model(torch.randn(1, 10))


def test_configurable_model_forward_implementation():
    """Test that ConfigurableModel's forward method correctly processes input through steps."""
    model = ConfigurableModel()
    input_tensor = torch.tensor([1.0, 2.0, 3.0])

    # Test with no steps added
    output = model(input_tensor)
    assert torch.allclose(output, input_tensor), "Forward should return input unchanged when no steps exist"

    # Test with empty steps list
    model.steps = []
    output = model(input_tensor)
    assert torch.allclose(output, input_tensor), "Forward should return input unchanged with empty steps list"


def test_configurable_model_forward_with_kwargs():
    """Test that ConfigurableModel's forward method handles kwargs correctly."""
    model = ConfigurableModel()

    def step_with_kwargs(x, scale=1):
        return x * scale

    model.add_step(step_with_kwargs)
    input_tensor = torch.tensor([1.0, 2.0, 3.0])

    # Test with default kwargs
    output = model(input_tensor)
    assert torch.allclose(output, input_tensor), "Should use default kwargs value"

    # Test with custom kwargs
    output = model(input_tensor, scale=2)
    assert torch.allclose(output, input_tensor * 2), "Should use provided kwargs value"


# Tests for ChannelAwareBaseModel
class DummyChannelAwareModel(ChannelAwareBaseModel):
    """Simple implementation for testing."""

    def forward(self, x: torch.Tensor, csi: torch.Tensor, **kwargs) -> torch.Tensor:
        # Simple operation that uses both input and CSI
        # Handle broadcasting for different input shapes
        csi_mean = csi.mean(dim=-1, keepdim=True)
        if x.dim() > csi_mean.dim():
            # Expand CSI mean to match input dimensions
            for _ in range(x.dim() - csi_mean.dim()):
                csi_mean = csi_mean.unsqueeze(-1)
        return x + csi_mean


class IncompleteChannelAwareModel(ChannelAwareBaseModel):
    """Model that doesn't implement forward method properly."""

    pass


def test_channel_aware_base_model_abstract():
    """Test that ChannelAwareBaseModel cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ChannelAwareBaseModel()


def test_channel_aware_base_model_incomplete_subclass():
    """Test that subclasses must implement the forward method."""
    with pytest.raises(TypeError):
        IncompleteChannelAwareModel()


def test_channel_aware_base_model_forward():
    """Test basic forward pass with CSI."""
    model = DummyChannelAwareModel()
    x = torch.randn(4, 16, 32, 32)
    csi = torch.randn(4, 8)

    output = model(x, csi=csi)

    # Check output shape matches input
    assert output.shape == x.shape
    # Check that CSI was actually used (output should differ from input)
    assert not torch.allclose(output, x)


def test_channel_aware_base_model_forward_missing_csi():
    """Test that forward method requires CSI parameter."""
    model = DummyChannelAwareModel()
    x = torch.randn(4, 16, 32, 32)

    with pytest.raises(TypeError):
        model(x)  # Missing required csi parameter


def test_validate_csi_valid():
    """Test CSI validation with valid tensors."""
    model = DummyChannelAwareModel()

    # Valid real CSI
    csi_real = torch.randn(4, 8)
    validated = model.validate_csi(csi_real)
    assert torch.allclose(validated, csi_real)

    # Valid complex CSI
    csi_complex = torch.complex(torch.randn(4, 8), torch.randn(4, 8))
    validated = model.validate_csi(csi_complex)
    assert torch.allclose(validated, csi_complex)


def test_validate_csi_invalid():
    """Test CSI validation with invalid inputs."""
    model = DummyChannelAwareModel()

    # Test with non-tensor input
    with pytest.raises(TypeError):
        model.validate_csi([1, 2, 3])

    # Test with tensor containing NaN
    csi_nan = torch.tensor([1.0, float("nan"), 3.0])
    with pytest.raises(ValueError):
        model.validate_csi(csi_nan)

    # Test with tensor containing Inf
    csi_inf = torch.tensor([1.0, float("inf"), 3.0])
    with pytest.raises(ValueError):
        model.validate_csi(csi_inf)


def test_normalize_csi_minmax():
    """Test CSI normalization using minmax scaling."""
    model = DummyChannelAwareModel()
    csi = torch.tensor([[1.0, 5.0, 3.0], [0.0, 10.0, 2.0]])

    normalized = model.normalize_csi(csi, method="minmax")

    # Check that values are in [0, 1] range
    assert torch.all(normalized >= 0)
    assert torch.all(normalized <= 1)

    # Check that normalization is applied per batch row
    # For row 0: [1, 5, 3] -> [0, 1, 0.5] (min=1, max=5)
    # For row 1: [0, 10, 2] -> [0, 1, 0.2] (min=0, max=10)
    expected = torch.tensor([[0.0, 1.0, 0.5], [0.0, 1.0, 0.2]])
    assert torch.allclose(normalized, expected)


def test_normalize_csi_zscore():
    """Test CSI normalization using z-score scaling."""
    model = DummyChannelAwareModel()
    # Use a deterministic input for more predictable results
    csi = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])  # Single batch with mean=2, std=1.58

    normalized = model.normalize_csi(csi, method="zscore")

    # Check that mean is approximately 0 and std is approximately 1 for the batch dimension
    batch_mean = normalized.mean(dim=1)
    batch_std = normalized.std(dim=1, unbiased=False)  # Use population std

    assert torch.allclose(batch_mean, torch.zeros(1), atol=1e-6)
    assert torch.allclose(batch_std, torch.ones(1), atol=1e-6)


def test_normalize_csi_none():
    """Test CSI normalization with no normalization."""
    model = DummyChannelAwareModel()
    csi = torch.randn(4, 8)

    normalized = model.normalize_csi(csi, method="none")

    # Should return the same tensor
    assert torch.allclose(normalized, csi)


def test_normalize_csi_invalid_method():
    """Test CSI normalization with invalid method."""
    model = DummyChannelAwareModel()
    csi = torch.randn(4, 8)

    with pytest.raises(ValueError):
        model.normalize_csi(csi, method="invalid")


def test_transform_csi_reshape():
    """Test CSI transformation by reshaping."""
    model = DummyChannelAwareModel()
    csi = torch.randn(4, 8)

    # Reshape to 2D
    transformed = model.transform_csi(csi, target_shape=(4, 2, 4))
    assert transformed.shape == (4, 2, 4)
    assert torch.allclose(transformed.flatten(start_dim=1), csi)

    # Reshape to 3D
    transformed = model.transform_csi(csi, target_shape=(4, 2, 2, 2))
    assert transformed.shape == (4, 2, 2, 2)


def test_transform_csi_invalid_shape():
    """Test CSI transformation with incompatible shape."""
    model = DummyChannelAwareModel()
    csi = torch.randn(4, 8)  # 32 elements total

    # Target shape has more elements - should pad
    transformed = model.transform_csi(csi, target_shape=(4, 10))  # 40 elements
    assert transformed.shape == (4, 10)

    # Target shape has fewer elements - should truncate
    transformed = model.transform_csi(csi, target_shape=(4, 6))  # 24 elements
    assert transformed.shape == (4, 6)


def test_extract_csi_features():
    """Test extraction of statistical features from CSI."""
    model = DummyChannelAwareModel()

    # Create CSI with known statistics
    csi = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

    # The actual method doesn't take a features parameter
    features = model.extract_csi_features(csi)

    assert "mean" in features
    assert "std" in features
    assert "min" in features
    assert "max" in features

    # Check values
    assert torch.allclose(features["mean"], torch.tensor(3.0))
    assert torch.allclose(features["min"], torch.tensor(1.0))
    assert torch.allclose(features["max"], torch.tensor(5.0))


def test_extract_csi_features_complex():
    """Test extraction of features from complex CSI."""
    model = DummyChannelAwareModel()

    # Complex CSI
    csi = torch.complex(torch.randn(4, 8), torch.randn(4, 8))

    # The method extracts standard features, including magnitude and phase
    features = model.extract_csi_features(csi)

    assert "magnitude" in features
    assert "phase" in features
    assert features["magnitude"].shape == ()  # scalar
    assert features["phase"].shape == ()  # scalar


def test_extract_csi_features_invalid():
    """Test extraction with real CSI (not testing invalid features since method doesn't take
    features param)."""
    model = DummyChannelAwareModel()
    csi = torch.randn(4, 8)

    # This should work fine - testing that the method handles real CSI properly
    features = model.extract_csi_features(csi)

    # Should contain basic statistical features
    assert "mean" in features
    assert "std" in features
    assert "min" in features
    assert "max" in features


def test_forward_csi_to_submodules():
    """Test helper for forwarding CSI to submodules."""
    model = DummyChannelAwareModel()

    # Create mock submodules
    submodule1 = DummyChannelAwareModel()
    submodule2 = DummyChannelAwareModel()
    submodules = [submodule1, submodule2]

    x = torch.randn(4, 16)
    csi = torch.randn(4, 8)

    outputs = model.forward_csi_to_submodules(csi, submodules, x)

    assert len(outputs) == 2
    assert all(output.shape == x.shape for output in outputs)


def test_create_csi_for_submodules():
    """Test helper for creating CSI copies for submodules."""
    model = DummyChannelAwareModel()
    csi = torch.randn(4, 8)

    csi_copies = model.create_csi_for_submodules(csi, num_modules=3)

    assert len(csi_copies) == 3
    assert all(torch.allclose(copy, csi) for copy in csi_copies)


def test_extract_csi_from_channel_output():
    """Test static method for extracting CSI from channel outputs."""
    # Test with tuple output (signal, csi)
    signal = torch.randn(4, 16)
    csi = torch.randn(4, 8)
    channel_output = (signal, csi)

    extracted_csi = ChannelAwareBaseModel.extract_csi_from_channel_output(channel_output)
    assert torch.allclose(extracted_csi, csi)

    # Test with dict output
    channel_output = {"signal": signal, "csi": csi, "other": torch.randn(4, 4)}
    extracted_csi = ChannelAwareBaseModel.extract_csi_from_channel_output(channel_output)
    assert torch.allclose(extracted_csi, csi)


def test_extract_csi_from_channel_output_no_csi():
    """Test CSI extraction when no CSI is present."""
    signal = torch.randn(4, 16)

    # Test with single tensor (no CSI)
    extracted_csi = ChannelAwareBaseModel.extract_csi_from_channel_output(signal)
    assert extracted_csi is None

    # Test with dict without CSI key
    channel_output = {"signal": signal, "other": torch.randn(4, 4)}
    extracted_csi = ChannelAwareBaseModel.extract_csi_from_channel_output(channel_output)
    assert extracted_csi is None


def test_format_csi_for_channel():
    """Test static method for formatting CSI for channel input."""
    csi = torch.randn(4, 8)

    # Test tensor format (default)
    formatted = ChannelAwareBaseModel.format_csi_for_channel(csi, channel_format="tensor")
    assert torch.allclose(formatted, csi)

    # Test dict format
    formatted = ChannelAwareBaseModel.format_csi_for_channel(csi, channel_format="dict")
    assert isinstance(formatted, dict)
    assert "csi" in formatted
    assert torch.allclose(formatted["csi"], csi)


def test_format_csi_for_channel_invalid():
    """Test CSI formatting with invalid format type."""
    csi = torch.randn(4, 8)

    with pytest.raises(ValueError):
        ChannelAwareBaseModel.format_csi_for_channel(csi, channel_format="invalid")


@pytest.mark.parametrize("batch_size,csi_length", [(1, 4), (8, 16), (32, 64)])
def test_channel_aware_model_different_sizes(batch_size, csi_length):
    """Test ChannelAwareBaseModel with different batch sizes and CSI lengths."""
    model = DummyChannelAwareModel()
    x = torch.randn(batch_size, 16)
    csi = torch.randn(batch_size, csi_length)

    output = model(x, csi=csi)
    assert output.shape[0] == batch_size
    assert output.shape[1:] == x.shape[1:]


def test_channel_aware_model_device_compatibility():
    """Test that ChannelAwareBaseModel works with different devices."""
    model = DummyChannelAwareModel()
    x = torch.randn(4, 16)
    csi = torch.randn(4, 8)

    # Test on CPU
    output_cpu = model(x, csi=csi)
    assert output_cpu.device == x.device

    # Test GPU compatibility if available
    if torch.cuda.is_available():
        model_gpu = model.cuda()
        x_gpu = x.cuda()
        csi_gpu = csi.cuda()

        output_gpu = model_gpu(x_gpu, csi=csi_gpu)
        assert output_gpu.device == x_gpu.device


def test_channel_aware_model_dtype_compatibility():
    """Test that ChannelAwareBaseModel preserves data types."""
    model = DummyChannelAwareModel()

    # Test with float32
    x_float32 = torch.randn(4, 16, dtype=torch.float32)
    csi_float32 = torch.randn(4, 8, dtype=torch.float32)
    output = model(x_float32, csi=csi_float32)
    assert output.dtype == torch.float32

    # Test with float64
    x_float64 = torch.randn(4, 16, dtype=torch.float64)
    csi_float64 = torch.randn(4, 8, dtype=torch.float64)
    output = model(x_float64, csi=csi_float64)
    assert output.dtype == torch.float64
