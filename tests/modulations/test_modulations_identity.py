import torch

from kaira.modulations import IdentityModulator  # Changed from IdentityModulation


def test_identity_modulation_modulate():
    """Test that identity modulator returns input unchanged."""
    modulator = IdentityModulator()  # Changed class name
    input_signal = torch.tensor([1, 2, 3, 4, 5])
    modulated_signal = modulator(input_signal)  # Using __call__ instead of modulate
    assert torch.equal(modulated_signal, input_signal)


def test_identity_modulation_forward():
    """Test forward method of identity modulator."""
    modulator = IdentityModulator()
    input_signal = torch.tensor([1, 2, 3, 4, 5])
    modulated_signal = modulator.forward(input_signal)
    assert torch.equal(modulated_signal, input_signal)


def test_identity_modulation_empty():
    """Test identity modulator with empty tensor."""
    modulator = IdentityModulator()
    input_signal = torch.tensor([])
    modulated_signal = modulator(input_signal)
    assert torch.equal(modulated_signal, input_signal)


def test_identity_modulation_bits_per_symbol():
    """Test bits_per_symbol property of identity modulator."""
    modulator = IdentityModulator()
    # If bits_per_symbol is implemented, test its value
    # If not implemented, test that it raises NotImplementedError
    try:
        bits_per_symbol = modulator.bits_per_symbol
        assert isinstance(bits_per_symbol, int)
    except NotImplementedError:
        pass  # This is also acceptable if not implemented


def test_identity_modulation_reset_state():
    """Test reset_state method of identity modulator."""
    modulator = IdentityModulator()
    # Should not raise exception
    modulator.reset_state()
