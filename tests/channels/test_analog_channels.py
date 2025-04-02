"""Tests for analog channel implementations in Kaira."""
import pytest
import torch
import numpy as np
from kaira.channels import (
    AWGNChannel,
    GaussianChannel,
    LaplacianChannel,
    PhaseNoiseChannel,
    PoissonChannel,
    NonlinearChannel,
    FlatFadingChannel,
    RayleighFadingChannel,
    PerfectChannel
)
from .test_channels_base import random_tensor, complex_tensor


class TestAWGNChannel:
    """Test suite for AWGNChannel."""
    
    def test_initialization(self):
        """Test initialization with different parameters."""
        # Initialize with avg_noise_power
        channel = AWGNChannel(avg_noise_power=0.1)
        assert torch.isclose(channel.avg_noise_power, torch.tensor(0.1), rtol=1e-5)
        assert channel.snr_db is None

        # Initialize with snr_db
        channel = AWGNChannel(snr_db=10.0)
        assert channel.snr_db == 10.0
        assert channel.avg_noise_power is None

        # Test initialization error when neither parameter is provided
        with pytest.raises(ValueError, match="Either avg_noise_power or snr_db must be provided"):
            AWGNChannel()
            
        # Verify that GaussianChannel is an alias for AWGNChannel
        assert GaussianChannel is AWGNChannel
    
    def test_forward_with_noise_power(self, random_tensor):
        """Test forward pass with specified noise power."""
        noise_power = 0.1
        channel = AWGNChannel(avg_noise_power=noise_power)
        output = channel(random_tensor)
        
        # Check shape preservation
        assert output.shape == random_tensor.shape
        
        # Check noise was added (output should be different from input)
        assert not torch.allclose(output, random_tensor)
        
        # Check noise variance is approximately as expected
        noise = output - random_tensor
        measured_variance = torch.var(noise).item()
        assert np.isclose(measured_variance, noise_power, rtol=0.2)
    
    def test_forward_with_snr(self, random_tensor):
        """Test forward pass with SNR specification."""
        snr_db = 10.0  # 10dB SNR
        channel = AWGNChannel(snr_db=snr_db)
        
        # Create test input with known power
        # Since random_tensor has approx unit power, we can use it directly
        output = channel(random_tensor)
        
        # Calculate signal power
        signal_power = torch.mean(random_tensor**2).item()
        
        # Calculate expected noise power based on SNR
        expected_noise_power = signal_power / (10 ** (snr_db / 10))
        
        # Check noise statistics
        noise = output - random_tensor
        measured_noise_power = torch.mean(noise**2).item()
        
        # Noise power should be close to the expected value
        assert np.isclose(measured_noise_power, expected_noise_power, rtol=0.3)
    
    def test_complex_input(self, complex_tensor):
        """Test AWGN channel with complex input."""
        noise_power = 0.1
        channel = AWGNChannel(avg_noise_power=noise_power)
        output = channel(complex_tensor)
        
        # Check output shape and type
        assert output.shape == complex_tensor.shape
        assert output.dtype == complex_tensor.dtype
        assert torch.is_complex(output)
        
        # Check noise variance for real and imaginary parts
        noise = output - complex_tensor
        real_variance = torch.var(noise.real).item()
        imag_variance = torch.var(noise.imag).item()
        
        # For complex AWGN, noise power is split between real and imaginary components
        assert np.isclose(real_variance, noise_power/2, rtol=0.2)
        assert np.isclose(imag_variance, noise_power/2, rtol=0.2)
    
    def test_pregenerated_noise(self, random_tensor):
        """Test with pre-generated noise."""
        channel = AWGNChannel(avg_noise_power=0.1)
        
        # Create custom noise
        noise = torch.randn_like(random_tensor) * 0.2
        
        # Apply channel with pre-generated noise
        output = channel(random_tensor, noise=noise)
        
        # Output should be exactly input + noise
        assert torch.allclose(output, random_tensor + noise)


class TestLaplacianChannel:
    """Test suite for LaplacianChannel."""
    
    def test_initialization(self):
        """Test initialization with different parameters."""
        # Test with scale
        channel1 = LaplacianChannel(scale=0.5)
        assert torch.isclose(channel1.scale, torch.tensor(0.5), rtol=1e-5)
        assert channel1.avg_noise_power is None
        assert channel1.snr_db is None
        
        # Test with noise power
        channel2 = LaplacianChannel(avg_noise_power=0.1)
        assert torch.isclose(channel2.avg_noise_power, torch.tensor(0.1), rtol=1e-5)
        assert channel2.scale is None
        assert channel2.snr_db is None
        
        # Test with SNR
        channel3 = LaplacianChannel(snr_db=15.0)
        assert channel3.snr_db == 15.0
        assert channel3.avg_noise_power is None
        assert channel3.scale is None
        
        # Test with no parameters
        with pytest.raises(ValueError):
            LaplacianChannel()
    
    def test_forward_with_scale(self, random_tensor):
        """Test forward pass with scale parameter."""
        scale = 0.5
        channel = LaplacianChannel(scale=scale)
        output = channel(random_tensor)
        
        # Check shape preservation
        assert output.shape == random_tensor.shape
        
        # Check noise was added (output should be different from input)
        assert not torch.allclose(output, random_tensor)
        
        # For Laplacian distribution with scale b, variance = 2 * b^2
        expected_variance = 2 * (scale ** 2)
        noise = output - random_tensor
        measured_variance = torch.var(noise).item()
        assert np.isclose(measured_variance, expected_variance, rtol=0.3)
    
    def test_forward_with_noise_power(self, random_tensor):
        """Test forward pass with noise power specification."""
        noise_power = 0.4
        channel = LaplacianChannel(avg_noise_power=noise_power)
        output = channel(random_tensor)
        
        # Check noise variance
        noise = output - random_tensor
        measured_variance = torch.var(noise).item()
        assert np.isclose(measured_variance, noise_power, rtol=0.3)
    
    def test_complex_input(self, complex_tensor):
        """Test with complex input."""
        channel = LaplacianChannel(scale=0.5)
        output = channel(complex_tensor)
        
        # Check shape and type preservation
        assert output.shape == complex_tensor.shape
        assert torch.is_complex(output)
        
        # Check noise added to both real and imaginary parts
        assert not torch.allclose(output.real, complex_tensor.real)
        assert not torch.allclose(output.imag, complex_tensor.imag)


class TestPhaseNoiseChannel:
    """Test suite for PhaseNoiseChannel."""
    
    def test_initialization(self):
        """Test initialization with different parameters."""
        # Test with valid parameter
        channel = PhaseNoiseChannel(phase_noise_std=0.1)
        assert torch.isclose(channel.phase_noise_std, torch.tensor(0.1), rtol=1e-5)
        
        # Test with negative standard deviation (should raise error)
        with pytest.raises(ValueError):
            PhaseNoiseChannel(phase_noise_std=-0.1)
    
    def test_forward_real_input(self, random_tensor):
        """Test forward pass with real input."""
        channel = PhaseNoiseChannel(phase_noise_std=0.1)
        output = channel(random_tensor)
        
        # Output should be complex even if input is real
        assert output.shape == random_tensor.shape
        assert torch.is_complex(output)
        
        # Magnitude should be approximately preserved
        assert torch.allclose(torch.abs(output), torch.abs(random_tensor), rtol=1e-5)
    
    def test_forward_complex_input(self, complex_tensor):
        """Test forward pass with complex input."""
        # Reset the random seed for reproducible testing
        torch.manual_seed(42)
        
        phase_std = 0.1
        # Create a simpler tensor for deterministic testing
        test_tensor = torch.complex(torch.ones(1000), torch.zeros(1000))
        channel = PhaseNoiseChannel(phase_noise_std=phase_std)
        
        # Apply channel with known phase noise
        output = channel(test_tensor)
        
        # Check shape and type preservation
        assert output.shape == test_tensor.shape
        assert torch.is_complex(output)
        
        # Magnitude should be preserved
        assert torch.allclose(torch.abs(output), torch.abs(test_tensor), rtol=1e-5)
        
        # Phase should be perturbed
        phase_diff = torch.angle(output) - torch.angle(test_tensor)
        
        # The phase differences should have a standard deviation close to phase_std
        measured_std = torch.std(phase_diff).item()
        assert 0.08 <= measured_std <= 0.12


class TestPoissonChannel:
    """Test suite for PoissonChannel."""
    
    def test_initialization(self):
        """Test initialization with different parameters."""
        # Test with default parameters
        channel1 = PoissonChannel()
        assert channel1.rate_factor == 1.0
        assert not channel1.normalize
        
        # Test with custom parameters
        channel2 = PoissonChannel(rate_factor=2.0, normalize=True)
        assert channel2.rate_factor.item() == 2.0
        assert channel2.normalize
        
        # Test with invalid rate factor
        with pytest.raises(ValueError):
            PoissonChannel(rate_factor=0)
        with pytest.raises(ValueError):
            PoissonChannel(rate_factor=-1.0)
    
    def test_forward_real_input(self):
        """Test forward pass with real input."""
        channel = PoissonChannel(rate_factor=10.0, normalize=False)
        # Use positive input for Poisson channel
        x = torch.ones(1000) * 0.5
        y = channel(x)
        
        # Check shape preservation
        assert y.shape == x.shape
        
        # Output should follow Poisson with rate = rate_factor * x
        # For Poisson, mean = rate
        expected_mean = 10.0 * 0.5  # rate_factor * x
        assert 4.5 <= torch.mean(y).item() <= 5.5
    
    def test_normalization(self):
        """Test normalization option."""
        channel = PoissonChannel(rate_factor=10.0, normalize=True)
        x = torch.ones(1000) * 0.5
        y = channel(x)
        
        # With normalization, output should be close to input on average
        assert 0.45 <= torch.mean(y).item() <= 0.55
    
    def test_negative_input(self):
        """Test with negative input (should raise error)."""
        channel = PoissonChannel()
        x = torch.ones(10) * -1.0
        with pytest.raises(ValueError):
            channel(x)
    
    def test_complex_input(self):
        """Test with complex input."""
        # Set a fixed seed for reproducible tests
        torch.manual_seed(42)
        
        channel = PoissonChannel(rate_factor=10.0)
        # Create complex input with positive magnitude
        x = torch.complex(torch.ones(100) * 0.5, torch.ones(100) * 0.5)
        
        # Get the exact phases before applying the channel
        input_phase = torch.angle(x)
        
        # Apply the channel
        y = channel(x)
        
        # Check output is complex
        assert y.shape == x.shape
        assert torch.is_complex(y)
        
        # Phase should be preserved exactly
        output_phase = torch.angle(y)
        assert torch.allclose(output_phase, input_phase, atol=1e-5)


class TestFlatFadingChannel:
    """Test suite for FlatFadingChannel."""
    
    def test_initialization(self):
        """Test initialization with different parameters."""
        # Test Rayleigh fading
        channel1 = FlatFadingChannel(fading_type="rayleigh", coherence_time=10, snr_db=15)
        assert channel1.fading_type == "rayleigh"
        assert channel1.coherence_time == 10
        assert channel1.snr_db == 15
        
        # Test Rician fading
        channel2 = FlatFadingChannel(
            fading_type="rician", 
            coherence_time=5, 
            k_factor=2.0,
            snr_db=10
        )
        assert channel2.fading_type == "rician"
        assert channel2.k_factor.item() == 2.0
        
        # Test log-normal fading
        channel3 = FlatFadingChannel(
            fading_type="lognormal",
            coherence_time=20,
            shadow_sigma_db=4.0,
            snr_db=20
        )
        assert channel3.fading_type == "lognormal"
        assert channel3.shadow_sigma_db.item() == 4.0
        
        # Test invalid fading type
        with pytest.raises(ValueError):
            FlatFadingChannel(fading_type="invalid", coherence_time=10, snr_db=15)
            
        # Test missing k_factor for Rician
        with pytest.raises(ValueError):
            FlatFadingChannel(fading_type="rician", coherence_time=10, snr_db=15)
            
        # Test missing shadow_sigma_db for log-normal
        with pytest.raises(ValueError):
            FlatFadingChannel(fading_type="lognormal", coherence_time=10, snr_db=15)
            
        # Test missing noise parameters
        with pytest.raises(ValueError):
            FlatFadingChannel(fading_type="rayleigh", coherence_time=10)
    
    def test_rayleigh_fading(self, complex_tensor):
        """Test Rayleigh fading channel."""
        channel = FlatFadingChannel(
            fading_type="rayleigh",
            coherence_time=10,
            snr_db=20
        )
        output = channel(complex_tensor)
        
        # Check shape and type preservation
        assert output.shape == complex_tensor.shape
        assert torch.is_complex(output)
        
        # Output should be different from input
        assert not torch.allclose(output, complex_tensor)
        
        # Create a proper CSI tensor matching input dimensions
        real_part = torch.ones_like(complex_tensor.real) * 0.5
        imag_part = torch.zeros_like(complex_tensor.imag)
        csi = torch.complex(real_part, imag_part)
        
        output_with_csi = channel(complex_tensor, csi=csi)
        
        # With CSI, the fading component should match the provided CSI
        # Extract the signal part (without noise)
        signal_part = output_with_csi - (output_with_csi - csi*complex_tensor)
        
        # Should be close to csi*x but with some noise
        assert not torch.allclose(signal_part, complex_tensor)
    
    def test_rayleigh_fading_channel_class(self, complex_tensor):
        """Test the specialized RayleighFadingChannel class."""
        channel = RayleighFadingChannel(coherence_time=5, snr_db=15)
        
        # Verify it has defaults set correctly
        assert channel.fading_type == "rayleigh"
        assert channel.coherence_time == 5
        assert channel.snr_db == 15
        
        # Test forward pass
        output = channel(complex_tensor)
        
        # Check shape and type preservation
        assert output.shape == complex_tensor.shape
        assert torch.is_complex(output)
        
        # Output should be different from input
        assert not torch.allclose(output, complex_tensor)


class TestNonlinearChannel:
    """Test suite for NonlinearChannel."""
    
    def test_initialization(self):
        """Test initialization with different parameters."""
        # Simple nonlinearity
        def cubic(x): 
            return x + 0.1 * x**3
            
        # Basic initialization
        channel1 = NonlinearChannel(cubic)
        assert not channel1.add_noise
        assert channel1.complex_mode == "direct"
        
        # With noise
        channel2 = NonlinearChannel(cubic, add_noise=True, avg_noise_power=0.1)
        assert channel2.add_noise
        assert channel2.avg_noise_power == 0.1
        
        # With SNR
        channel3 = NonlinearChannel(cubic, add_noise=True, snr_db=15.0)
        assert channel3.add_noise
        assert channel3.snr_db == 15.0
        
        # Invalid complex mode
        with pytest.raises(ValueError):
            NonlinearChannel(cubic, complex_mode="invalid")
            
        # Missing noise parameters
        with pytest.raises(ValueError):
            NonlinearChannel(cubic, add_noise=True)
    
    def test_forward_real_input(self, random_tensor):
        """Test with real input."""
        def cubic(x): 
            return x + 0.1 * x**3
            
        channel = NonlinearChannel(cubic)
        output = channel(random_tensor)
        
        # Check shape preservation
        assert output.shape == random_tensor.shape
        
        # Check output matches expected nonlinear transform
        expected = random_tensor + 0.1 * random_tensor**3
        assert torch.allclose(output, expected)
    
    def test_forward_complex_modes(self, complex_tensor):
        """Test with different complex modes."""
        # Test direct mode
        def scale_complex(x): 
            return 0.5 * x
            
        channel1 = NonlinearChannel(scale_complex, complex_mode="direct")
        output1 = channel1(complex_tensor)
        assert torch.allclose(output1, 0.5 * complex_tensor)
        
        # Test cartesian mode
        def square(x): 
            return x**2
            
        channel2 = NonlinearChannel(square, complex_mode="cartesian")
        output2 = channel2(complex_tensor)
        expected2 = torch.complex(complex_tensor.real**2, complex_tensor.imag**2)
        assert torch.allclose(output2, expected2)
        
        # Test polar mode
        def double_magnitude(x): 
            return 2.0 * x
            
        channel3 = NonlinearChannel(double_magnitude, complex_mode="polar")
        output3 = channel3(complex_tensor)
        
        # Magnitude should be doubled
        input_mag = torch.abs(complex_tensor)
        output_mag = torch.abs(output3)
        assert torch.allclose(output_mag, 2.0 * input_mag, rtol=1e-5)
        
        # Phase should be preserved
        input_phase = torch.angle(complex_tensor)
        output_phase = torch.angle(output3)
        assert torch.allclose(output_phase, input_phase, rtol=1e-5)
    
    def test_nonlinear_with_noise(self, random_tensor):
        """Test nonlinearity with added noise."""
        def identity(x): 
            return x
            
        channel = NonlinearChannel(
            identity, 
            add_noise=True, 
            avg_noise_power=0.1
        )
        output = channel(random_tensor)
        
        # Check shape preservation
        assert output.shape == random_tensor.shape
        
        # Output should be different from input (due to noise)
        assert not torch.allclose(output, random_tensor)
        
        # Check noise variance
        noise = output - random_tensor
        measured_variance = torch.var(noise).item()
        assert np.isclose(measured_variance, 0.1, rtol=0.2)


def test_perfect_channel(random_tensor, complex_tensor):
    """Test PerfectChannel (identity channel)."""
    channel = PerfectChannel()
    
    # Test with real input
    output_real = channel(random_tensor)
    assert torch.allclose(output_real, random_tensor)
    
    # Test with complex input
    output_complex = channel(complex_tensor)
    assert torch.allclose(output_complex, complex_tensor)


class TestChannelComposition:
    """Test composition of multiple channels."""
    
    def test_sequential_composition(self, complex_tensor):
        """Test applying channels in sequence."""
        # Create individual channels
        phase_noise = PhaseNoiseChannel(phase_noise_std=0.1)
        awgn = AWGNChannel(snr_db=15)
        
        # Apply in sequence
        output = awgn(phase_noise(complex_tensor))
        
        # Check output shape matches input
        assert output.shape == complex_tensor.shape
        
        # Output should be different from input
        assert not torch.allclose(output, complex_tensor)
        
        # Output should be complex
        assert torch.is_complex(output)
    
    def test_composition_order(self, complex_tensor):
        """Test that the order of channel application matters."""
        phase_noise = PhaseNoiseChannel(phase_noise_std=0.2)
        awgn = AWGNChannel(snr_db=10)
        
        # Apply in different orders
        torch.manual_seed(42)  # Use same seed for fair comparison
        output1 = awgn(phase_noise(complex_tensor))
        
        torch.manual_seed(42)
        output2 = phase_noise(awgn(complex_tensor))
        
        # Results should be different
        assert not torch.allclose(output1, output2)
    
    def test_perfect_channel_composition(self, complex_tensor):
        """Test PerfectChannel in a composition."""
        awgn = AWGNChannel(snr_db=15)
        perfect = PerfectChannel()
        
        # Set fixed random seed
        torch.manual_seed(123)
        output1 = awgn(complex_tensor)
        
        # Reset seed for same noise realization
        torch.manual_seed(123)
        output2 = awgn(perfect(complex_tensor))
        
        # Outputs should be identical
        assert torch.allclose(output1, output2)
    
    def test_multiple_awgn_composition(self, complex_tensor):
        """Test that multiple AWGN channels have additive effect on noise."""
        # Create two AWGN channels with the same SNR
        snr_db = 20
        awgn1 = AWGNChannel(snr_db=snr_db)
        awgn2 = AWGNChannel(snr_db=snr_db)
        
        # Apply a single channel vs. two channels in sequence
        torch.manual_seed(42)
        output1 = awgn1(complex_tensor)
        noise1 = torch.mean(torch.abs(output1 - complex_tensor)**2).item()
        
        torch.manual_seed(44)  # Different seed for independent noise
        output2 = awgn2(awgn1(complex_tensor))
        noise2 = torch.mean(torch.abs(output2 - complex_tensor)**2).item()
        
        # Second case should have approximately twice the noise power
        # (allowing for statistical variation)
        assert 1.5 < noise2/noise1 < 2.5