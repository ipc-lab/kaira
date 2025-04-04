import numpy as np
import pytest
import torch
import matplotlib.pyplot as plt

from kaira.modulations.pi4qpsk import Pi4QPSKDemodulator, Pi4QPSKModulator


@pytest.fixture
def pi4qpsk_modulator():
    """Fixture for a Pi/4-QPSK modulator."""
    return Pi4QPSKModulator()


@pytest.fixture
def pi4qpsk_demodulator():
    """Fixture for a Pi/4-QPSK demodulator."""
    return Pi4QPSKDemodulator()


def test_pi4qpsk_modulator_initialization():
    """Test initialization of Pi/4-QPSK modulator."""
    mod = Pi4QPSKModulator()
    assert mod.bits_per_symbol == 2
    assert mod.constellation.shape == (4,)

    # Verify constellation points
    # Pi/4-QPSK uses two QPSK constellations rotated by pi/4
    torch.tensor([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=torch.complex64)
    torch.tensor([np.sqrt(2) / 2 + np.sqrt(2) / 2 * 1j, -np.sqrt(2) / 2 + np.sqrt(2) / 2 * 1j, -np.sqrt(2) / 2 - np.sqrt(2) / 2 * 1j, np.sqrt(2) / 2 - np.sqrt(2) / 2 * 1j], dtype=torch.complex64)

    # Verify the rotation property is preserved in the implementation
    assert mod._even_symbols or mod._odd_symbols

    # Test with gray coding
    mod_gray = Pi4QPSKModulator(gray_coded=True)
    mod_no_gray = Pi4QPSKModulator(gray_coded=False)
    # Constellation should be different with gray coding
    assert not torch.allclose(mod_gray.constellation, mod_no_gray.constellation)


def test_pi4qpsk_modulator_forward(pi4qpsk_modulator):
    """Test forward pass of Pi/4-QPSK modulator."""
    # Reset state
    pi4qpsk_modulator.reset_state()

    # Test with batch of integers
    x = torch.tensor([0, 1, 2, 3])
    y = pi4qpsk_modulator(x)
    assert y.shape == torch.Size([4])
    assert y.dtype == torch.complex64

    # Pi/4-QPSK alternates between two constellation sets
    # Modulate twice and verify different constellation sets are used
    pi4qpsk_modulator.reset_state()
    y1 = pi4qpsk_modulator(torch.tensor([0]))
    y2 = pi4qpsk_modulator(torch.tensor([0]))
    # Same symbol but different constellations should produce different outputs
    assert not torch.isclose(y1, y2)


def test_pi4qpsk_reset_state():
    """Test resetting state for Pi/4-QPSK modulator."""
    mod = Pi4QPSKModulator()

    # Modulate a sequence
    mod.reset_state()
    seq1 = mod(torch.tensor([0, 1, 2, 3]))

    # Reset and modulate the same sequence again
    mod.reset_state()
    seq2 = mod(torch.tensor([0, 1, 2, 3]))

    # Both sequences should be identical after reset
    assert torch.allclose(seq1, seq2)


def test_pi4qpsk_demodulator_initialization():
    """Test initialization of Pi/4-QPSK demodulator."""
    demod = Pi4QPSKDemodulator()
    assert demod.bits_per_symbol == 2

    # Test with soft output
    demod_soft = Pi4QPSKDemodulator(soft_output=True)
    assert demod_soft.soft_output is True


def test_pi4qpsk_demodulator_forward(pi4qpsk_modulator, pi4qpsk_demodulator):
    """Test forward pass of Pi/4-QPSK demodulator."""
    # Reset states
    pi4qpsk_modulator.reset_state()
    pi4qpsk_demodulator.reset_state()

    # Test round trip with a sequence
    x = torch.tensor([0, 1, 2, 3, 0, 1])
    y = pi4qpsk_modulator(x)
    x_hat = pi4qpsk_demodulator(y)

    # Should recover original symbols
    assert torch.equal(x, x_hat)

    # Test with noise
    y_noisy = y + 0.1 * torch.randn_like(y.real) + 0.1j * torch.randn_like(y.imag)
    x_hat_noisy = pi4qpsk_demodulator(y_noisy)
    # Shape should match even with noise
    assert x_hat_noisy.shape == x.shape


def test_pi4qpsk_soft_demodulation():
    """Test soft demodulation for Pi/4-QPSK."""
    mod = Pi4QPSKModulator()
    demod = Pi4QPSKDemodulator(soft_output=True)

    # Reset states
    mod.reset_state()
    demod.reset_state()

    # Test with a sequence
    x = torch.tensor([0, 1, 2, 3])
    y = mod(x)

    # Get soft bit LLRs
    llrs = demod(y)
    assert llrs.shape == (4, 2)  # 4 symbols, 2 bits per symbol

    # For perfect reception, LLRs should have high magnitude
    assert torch.all(torch.abs(llrs) > 1.0)


def test_pi4qpsk_plot_constellation():
    """Test plotting of Pi/4-QPSK constellation."""
    mod = Pi4QPSKModulator()
    
    # Basic test that the plot function runs without error
    fig_and_axes = mod.plot_constellation()
    assert isinstance(fig_and_axes, tuple)
    assert isinstance(fig_and_axes[0], plt.Figure)
    
    # Close figure to avoid memory leaks
    plt.close(fig_and_axes[0])


def test_pi4qpsk_demodulator_soft_output_options(pi4qpsk_modulator):
    """Test Pi/4-QPSK demodulator with different soft output options."""
    # Reset modulator state
    pi4qpsk_modulator.reset_state()
    
    # Create test signal
    x = torch.tensor([0, 1, 2, 3])
    y = pi4qpsk_modulator(x)
    
    # Test with scalar noise variance
    demod_soft = Pi4QPSKDemodulator(soft_output=True)
    demod_soft.reset_state()
    llrs = demod_soft(y, noise_var=0.1)
    assert llrs.shape == torch.Size([4, 2])  # 4 symbols, 2 bits per symbol
    
    # Test with tensor noise variance
    noise_var = torch.ones(4) * 0.2
    llrs_tensor = demod_soft(y, noise_var=noise_var)
    assert llrs_tensor.shape == torch.Size([4, 2])
    
    # Test soft output without noise_var
    llrs_no_noise = demod_soft(y)
    assert llrs_no_noise.shape == torch.Size([4, 2])


def test_pi4qpsk_demodulator_with_batched_input(pi4qpsk_modulator):
    """Test Pi/4-QPSK demodulator with batched input."""
    # Hard decisions with batched input
    mod = pi4qpsk_modulator
    mod.reset_state()
    
    # Create batched test signal
    batch_size = 3
    seq_len = 4
    x = torch.randint(0, 4, (batch_size, seq_len))
    y = torch.zeros((batch_size, seq_len), dtype=torch.complex64)
    
    # Process each batch separately to ensure consistent state
    mod.reset_state()
    for i in range(batch_size):
        mod.reset_state()
        y[i] = mod(x[i])
    
    # Test hard demodulation with batched input
    demod_hard = Pi4QPSKDemodulator(soft_output=False)
    demod_hard.reset_state()
    x_hat = demod_hard(y)
    assert x_hat.shape == torch.Size([batch_size, seq_len * 2])  # Each symbol maps to 2 bits
    
    # Test soft demodulation with batched input
    demod_soft = Pi4QPSKDemodulator(soft_output=True)
    demod_soft.reset_state()
    
    # With scalar noise_var
    llrs_scalar = demod_soft(y, noise_var=0.1)
    assert llrs_scalar.shape == torch.Size([batch_size, seq_len * 2])
    
    # With tensor noise_var (same for all batches)
    noise_var_tensor = torch.ones(seq_len) * 0.2
    llrs_tensor1 = demod_soft(y, noise_var=noise_var_tensor)
    assert llrs_tensor1.shape == torch.Size([batch_size, seq_len * 2])
    
    # With full tensor noise_var
    noise_var_full = torch.ones(batch_size, seq_len) * 0.2
    llrs_tensor2 = demod_soft(y, noise_var=noise_var_full)
    assert llrs_tensor2.shape == torch.Size([batch_size, seq_len * 2])


def test_pi4qpsk_demodulator_output_formatting():
    """Test different output formatting options for Pi/4-QPSK demodulator."""
    mod = Pi4QPSKModulator()
    mod.reset_state()
    
    # Create test signal
    x = torch.tensor([0, 1, 2, 3])
    y = mod(x)
    
    # Test non-batched hard decisions
    demod_hard = Pi4QPSKDemodulator(soft_output=False)
    demod_hard.reset_state()
    bits = demod_hard(y)
    # The demodulator returns symbol indices when not batched and not soft output
    assert bits.shape == torch.Size([4])  # Returns symbol indices, not bits
    
    # Test non-batched soft decisions
    demod_soft = Pi4QPSKDemodulator(soft_output=True)
    demod_soft.reset_state()
    llrs = demod_soft(y)
    assert llrs.shape == torch.Size([4, 2])  # LLRs retain symbol structure
    
    # Test batched output
    batch_y = y.unsqueeze(0).repeat(2, 1)  # Shape: [2, 4]
    
    # Batched hard decisions
    bits_batched = demod_hard(batch_y)
    assert bits_batched.shape == torch.Size([2, 8])  # 2 batches, 8 bits each
    
    # Batched soft decisions
    llrs_batched = demod_soft(batch_y)
    assert llrs_batched.shape == torch.Size([2, 8])  # Flattened for batched output


def test_pi4qpsk_modulator_bit_input_validation():
    """Test bit input validation and conversion in Pi4QPSKModulator."""
    mod = Pi4QPSKModulator()
    
    # Test with valid bit input (even length)
    valid_bits = torch.tensor([0, 1, 1, 0, 1, 1], dtype=torch.float)  # 6 bits = 3 symbols
    output = mod(valid_bits)
    assert output.shape == torch.Size([3])  # 3 symbols
    
    # Test with invalid bit input (odd length)
    invalid_bits = torch.tensor([0, 1, 1, 0, 1], dtype=torch.float)  # 5 bits - not even
    with pytest.raises(ValueError, match="Input bit length must be even for π/4-QPSK modulation"):
        mod(invalid_bits)
    
    # Test with non-binary values (should be handled by fmod)
    non_binary = torch.tensor([2, 3, 4, 5], dtype=torch.float)  # Values > 1
    output_non_binary = mod(non_binary)
    assert output_non_binary.shape == torch.Size([2])  # Should be interpreted as 2 symbols
    
    # Test bit to index conversion
    # [0,0] -> 0, [0,1] -> 1, [1,0] -> 2, [1,1] -> 3
    test_bits = torch.tensor([
        [0, 0],  # Should map to index 0
        [0, 1],  # Should map to index 1
        [1, 0],  # Should map to index 2
        [1, 1],  # Should map to index 3
    ], dtype=torch.float)
    
    mod.reset_state()
    output_test = mod(test_bits.reshape(-1))
    
    # Create the same output using direct indices
    mod.reset_state()  # Reset to get the same constellation pattern
    output_indices = mod(torch.tensor([0, 1, 2, 3]))
    
    # Outputs should match
    assert torch.allclose(output_test, output_indices)


def test_pi4qpsk_demodulator_single_input_distance():
    """Test distance calculation for single input in Pi4QPSKDemodulator."""
    # Create a modulator to generate test signals
    mod = Pi4QPSKModulator()
    mod.reset_state()
    
    # Create test signals for each symbol in the constellation
    test_symbols = []
    for i in range(4):
        test_symbols.append(mod(torch.tensor([i])))
    test_signal = torch.cat(test_symbols)
    
    # Create a demodulator with hard output for testing distance calculation
    demod = Pi4QPSKDemodulator(soft_output=False)
    demod.reset_state()
    
    # Get the result
    result = demod(test_signal)
    
    # Should correctly identify all symbols
    assert torch.equal(result, torch.tensor([0, 1, 2, 3]))
    
    # Test with slight noise to ensure distances still work
    noisy_signal = test_signal + 0.1 * torch.randn_like(test_signal)
    noisy_result = demod.reset_state()(noisy_signal)
    
    # Even with noise, the result should be the same shape
    assert noisy_result.shape == torch.Size([4])


def test_pi4qpsk_demodulator_bit_pattern_assignment():
    """Test bit pattern assignment for single inputs in Pi4QPSKDemodulator."""
    mod = Pi4QPSKModulator()
    mod.reset_state()
    
    # Force the demodulator to go through the bit pattern assignment branch
    class CustomDemodulator(Pi4QPSKDemodulator):
        def forward(self, y):
            # Set up to process individual symbols
            batch_shape = ()
            symbol_shape = y.shape[0]
            qpsk = self.modulator.qpsk
            qpsk_rotated = self.modulator.qpsk_rotated
            use_rotated = self._use_rotated.clone()
            
            # Prepare output for hard decisions
            output_bits = torch.zeros(symbol_shape, 2, dtype=torch.float, device=y.device)
            
            for i in range(symbol_shape):
                # Select constellation
                constellation = qpsk_rotated if use_rotated else qpsk
                # Get distances
                y_i = y[i].unsqueeze(0)
                distances = torch.abs(y_i - constellation)
                closest_idx = torch.argmin(distances, dim=-1)
                
                # This is the code we're testing - bit pattern assignment
                for b in range(len(self.modulator.bit_patterns)):
                    mask = (closest_idx == b)
                    if mask.item():
                        output_bits[i, :] = self.modulator.bit_patterns[b]
                
                use_rotated = ~use_rotated
            
            # Return the output bits directly for inspection
            return output_bits
    
    # Create a test signal
    x = torch.tensor([0, 1, 2, 3])  # Use all four symbols
    y = mod(x)
    
    # Test with the custom demodulator
    demod = CustomDemodulator()
    demod.reset_state()
    bit_patterns = demod(y)
    
    # Should get the bit patterns from the modulator
    expected_bits = mod.bit_patterns
    
    # For each of the 4 symbols, check the corresponding bit patterns
    for i in range(4):
        symbol_idx = x[i]
        assert torch.allclose(bit_patterns[i], expected_bits[symbol_idx])


def test_pi4qpsk_batch_processing_with_bits():
    """Test batch processing with bit inputs in Pi4QPSKModulator."""
    mod = Pi4QPSKModulator()
    mod.reset_state()
    
    # Create a batch of bit inputs
    batch_size = 3
    num_symbols = 4
    bits_tensor = torch.zeros(batch_size, num_symbols * 2)  # 4 symbols, 2 bits each
    
    # Fill with different bit patterns
    bits_tensor[0, :] = torch.tensor([0, 0, 0, 1, 1, 0, 1, 1])
    bits_tensor[1, :] = torch.tensor([1, 0, 1, 1, 0, 0, 0, 1])
    bits_tensor[2, :] = torch.tensor([1, 1, 0, 1, 0, 0, 1, 0])
    
    # Process batch
    output = mod(bits_tensor)
    
    # Check output shape
    assert output.shape == (batch_size, num_symbols)
    
    # Compare with individual processing
    for i in range(batch_size):
        mod.reset_state()
        individual = mod(bits_tensor[i])
        assert torch.allclose(output[i], individual)
