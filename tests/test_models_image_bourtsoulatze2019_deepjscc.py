import pytest
import torch
from kaira.models.image.bourtsoulatze2019_deepjscc import Bourtsoulatze2019DeepJSCCEncoder, Bourtsoulatze2019DeepJSCCDecoder

@pytest.fixture
def encoder():
    return Bourtsoulatze2019DeepJSCCEncoder(num_transmitted_filters=64)

@pytest.fixture
def decoder():
    return Bourtsoulatze2019DeepJSCCDecoder(num_transmitted_filters=64)

def test_encoder_initialization(encoder):
    assert isinstance(encoder, Bourtsoulatze2019DeepJSCCEncoder)

def test_decoder_initialization(decoder):
    assert isinstance(decoder, Bourtsoulatze2019DeepJSCCDecoder)

def test_encoder_forward(encoder):
    input_tensor = torch.randn(4, 3, 32, 32)
    output = encoder(input_tensor)
    assert output.shape == (4, 64, 8, 8)

def test_decoder_forward(decoder):
    input_tensor = torch.randn(4, 64, 8, 8)
    output = decoder(input_tensor)
    assert output.shape == (4, 3, 32, 32)
