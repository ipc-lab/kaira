import pytest
import torch

from kaira.channels import AWGNChannel
from kaira.constraints import PeakAmplitudeConstraint
from kaira.models.deepjscc import DeepJSCCModel
from kaira.models.generic import SequentialModel
from kaira.models.registry import ModelRegistry


class SimpleEncoder(SequentialModel):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.layer(x)


class SimpleDecoder(SequentialModel):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(5, 10)

    def forward(self, x):
        return self.layer(x)


@pytest.fixture
def deepjscc_model():
    encoder = SimpleEncoder()
    constraint = PeakAmplitudeConstraint(max_amplitude=1.0)
    channel = AWGNChannel(avg_noise_power=0.1)
    decoder = SimpleDecoder()
    return DeepJSCCModel(encoder=encoder, constraint=constraint, channel=channel, decoder=decoder)


def test_deepjscc_model_initialization(deepjscc_model):
    assert isinstance(deepjscc_model, DeepJSCCModel)
    assert isinstance(deepjscc_model.encoder, SimpleEncoder)
    assert isinstance(deepjscc_model.constraint, PeakAmplitudeConstraint)
    assert isinstance(deepjscc_model.channel, AWGNChannel)
    assert isinstance(deepjscc_model.decoder, SimpleDecoder)


def test_deepjscc_model_forward(deepjscc_model):
    input_data = torch.randn(4, 10)
    output_data = deepjscc_model(input_data)
    assert output_data.shape == (4, 10)


def test_deepjscc_model_registry():
    assert "deepjscc" in ModelRegistry._models
    encoder = SimpleEncoder()
    constraint = PeakAmplitudeConstraint(max_amplitude=1.0)
    channel = AWGNChannel(avg_noise_power=0.1)
    decoder = SimpleDecoder()
    model = ModelRegistry.create("deepjscc", encoder=encoder, constraint=constraint, channel=channel, decoder=decoder)
    assert isinstance(model, DeepJSCCModel)
    assert model.encoder == encoder
    assert model.constraint == constraint
    assert model.channel == channel
    assert model.decoder == decoder


def test_deepjscc_model_with_kwargs(deepjscc_model):
    """Test DeepJSCCModel with additional keyword arguments."""
    input_data = torch.randn(4, 10)
    # Pass SNR as a kwarg that should be passed to the channel
    output_data = deepjscc_model(input_data, snr_db=15.0)
    assert output_data.shape == (4, 10)


def test_deepjscc_model_sequential_inheritance(deepjscc_model):
    """Test that DeepJSCCModel properly inherits from SequentialModel."""
    # DeepJSCCModel should have all the steps from the SequentialModel
    assert len(deepjscc_model.steps) == 4
    assert deepjscc_model.steps[0] == deepjscc_model.encoder
    assert deepjscc_model.steps[1] == deepjscc_model.constraint
    assert deepjscc_model.steps[2] == deepjscc_model.channel
    assert deepjscc_model.steps[3] == deepjscc_model.decoder


def test_deepjscc_model_device_compatibility(deepjscc_model):
    """Test DeepJSCCModel compatibility with different devices."""
    input_data = torch.randn(4, 10)

    # Move model to CPU explicitly
    deepjscc_model = deepjscc_model.to("cpu")
    input_data = input_data.to("cpu")

    # Forward pass should work on CPU
    output_cpu = deepjscc_model(input_data)
    assert output_cpu.device.type == "cpu"

    # Skip GPU test if not available
    if torch.cuda.is_available():
        # Move model to GPU
        deepjscc_model = deepjscc_model.to("cuda")
        input_data = input_data.to("cuda")

        # Forward pass should work on GPU
        output_gpu = deepjscc_model(input_data)
        assert output_gpu.device.type == "cuda"


def test_deepjscc_model_with_different_batch_sizes(deepjscc_model):
    """Test DeepJSCCModel with different batch sizes."""
    # Test with batch size 1
    input_data_single = torch.randn(1, 10)
    output_single = deepjscc_model(input_data_single)
    assert output_single.shape == (1, 10)

    # Test with a larger batch size
    input_data_large = torch.randn(16, 10)
    output_large = deepjscc_model(input_data_large)
    assert output_large.shape == (16, 10)


def test_deepjscc_model_gradient_flow(deepjscc_model):
    """Test gradient flow through the entire DeepJSCCModel."""
    input_data = torch.randn(4, 10, requires_grad=True)

    # Forward pass
    output = deepjscc_model(input_data)

    # Calculate loss
    loss = torch.mean(output)

    # Backward pass
    loss.backward()

    # Check that gradients flow through the model components
    for name, param in deepjscc_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
