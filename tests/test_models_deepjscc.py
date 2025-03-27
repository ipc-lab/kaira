import pytest
import torch
from kaira.models.deepjscc import DeepJSCCModel
from kaira.models.generic import SequentialModel
from kaira.channels import AWGNChannel
from kaira.constraints import PeakAmplitudeConstraint
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
    constraint = PeakAmplitudeConstraint(max_power=1.0)
    channel = AWGNChannel(avg_noise_power=0.1)
    decoder = SimpleDecoder()
    model = ModelRegistry.create(
        "deepjscc",
        encoder=encoder,
        constraint=constraint,
        channel=channel,
        decoder=decoder
    )
    assert isinstance(model, DeepJSCCModel)
    assert model.encoder == encoder
    assert model.constraint == constraint
    assert model.channel == channel
    assert model.decoder == decoder
