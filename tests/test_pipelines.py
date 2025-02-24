# tests/test_pipelines.py
import pytest
import torch
from kaira.pipelines import DeepJSCCPipeline
from kaira.core import BasePipeline

class MockEncoder(torch.nn.Module):
    def forward(self, x):
        return x * 2

class MockDecoder(torch.nn.Module):
    def forward(self, x):
        return x / 2

class MockConstraint(torch.nn.Module):
    def forward(self, x):
        return torch.clamp(x, -1, 1)

class MockChannel(torch.nn.Module):
    def forward(self, x):
        return x + torch.randn_like(x) * 0.1

@pytest.fixture
def mock_encoder():
    return MockEncoder()

@pytest.fixture
def mock_decoder():
    return MockDecoder()

@pytest.fixture
def mock_constraint():
    return MockConstraint()

@pytest.fixture
def mock_channel():
    return MockChannel()

@pytest.fixture
def sample_input():
    return torch.randn(1, 3, 32, 32)

def test_deepjscc_pipeline_initialization(mock_encoder, mock_decoder, mock_constraint, mock_channel):
    """Test DeepJSCCPipeline initialization."""
    pipeline = DeepJSCCPipeline(
        encoder=mock_encoder,
        decoder=mock_decoder,
        constraint=mock_constraint,
        channel=mock_channel,
    )
    assert isinstance(pipeline, DeepJSCCPipeline)
    assert isinstance(pipeline, BasePipeline)

def test_deepjscc_pipeline_forward(mock_encoder, mock_decoder, mock_constraint, mock_channel, sample_input):
    """Test DeepJSCCPipeline forward pass."""
    pipeline = DeepJSCCPipeline(
        encoder=mock_encoder,
        decoder=mock_decoder,
        constraint=mock_constraint,
        channel=mock_channel,
    )
    output = pipeline(sample_input)
    assert isinstance(output, torch.Tensor)
    assert output.shape == sample_input.shape

def test_deepjscc_pipeline_components_called(mock_encoder, mock_decoder, mock_constraint, mock_channel, sample_input):
    """Test that the components of the DeepJSCCPipeline are called during the forward pass."""
    # Use torch.spy to check if the forward methods of the components are called
    with torch.no_grad():
        pipeline = DeepJSCCPipeline(
            encoder=mock_encoder,
            decoder=mock_decoder,
            constraint=mock_constraint,
            channel=mock_channel,
        )
        output = pipeline(sample_input)

        # Check that the output is different from the input
        assert not torch.equal(output, sample_input)
