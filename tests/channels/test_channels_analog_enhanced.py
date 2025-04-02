"""Enhanced tests for the channels/analog.py module to increase coverage."""
import pytest
import torch
import numpy as np
from kaira.channels.analog import (
    _apply_noise,
    _to_complex,
    AWGNChannel,
    GaussianChannel,
    LaplacianChannel,
    PoissonChannel,
    PhaseNoiseChannel,
    FlatFadingChannel,
    NonlinearChannel,
    RayleighFadingChannel
)

class TestAnalogChannelUtils:
    """Test suite for analog channel utility functions."""
    
    def test_apply_noise_with_noise_power(self):
        """Test _apply_noise function with specified noise power."""
        x = torch.ones(100)
        y = _apply_noise(x, noise_power=0.1)
        
        assert y.shape == x.shape
        assert torch.mean((y - x) ** 2).item() > 0  # Should add noise
        # The variance of the noise should be close to the specified noise power
        assert 0.08 <= torch.var(y - x).item() <= 0.12  # Allow some tolerance
    
    def test_apply_noise_with_snr_db(self):
        """Test _apply_noise function with specified SNR in dB."""
        x = torch.ones(100) * 2.0  # Signal power = 4.0
        y = _apply_noise(x, snr_db=10.0)  # 10 dB SNR -> noise power = 0.4
        
        assert y.shape == x.shape
        # The variance of the noise should be close to the expected noise power
        assert 0.3 <= torch.var(y - x).item() <= 0.5  # Allow some tolerance
    
    def test_apply_noise_complex(self):
        """Test _apply_noise function with complex input."""
        x = torch.ones(100, dtype=torch.complex64)
        y = _apply_noise(x, noise_power=0.1)
        
        assert y.shape == x.shape
        assert torch.is_complex(y)
        # Check real and imaginary parts separately
        assert 0.04 <= torch.var(y.real - x.real).item() <= 0.06  # Noise power split between components
        assert 0.04 <= torch.var(y.imag - x.imag).item() <= 0.06
    
    def test_apply_noise_no_params(self):
        """Test _apply_noise function with no parameters."""
        x = torch.ones(100)
        with pytest.raises(ValueError):
            _apply_noise(x)  # Should raise error when no noise parameters provided
    
    def test_to_complex(self):
        """Test _to_complex function."""
        # Test with real input
        x_real = torch.ones(10)
        x_complex = _to_complex(x_real)
        assert torch.is_complex(x_complex)
        assert torch.allclose(x_complex.real, x_real)
        assert torch.allclose(x_complex.imag, torch.zeros_like(x_real))
        
        # Test with already complex input
        x_already_complex = torch.complex(torch.ones(10), torch.ones(10))
        x_result = _to_complex(x_already_complex)
        assert x_result is x_already_complex  # Should return the same object


class TestAWGNChannel:
    """Test suite for AWGNChannel."""
    
    def test_init_with_noise_power(self):
        """Test initialization with noise power."""
        channel = AWGNChannel(avg_noise_power=0.1)
        assert channel.avg_noise_power.item() == 0.1
        assert channel.snr_db is None
    
    def test_init_with_snr(self):
        """Test initialization with SNR."""
        channel = AWGNChannel(snr_db=15.0)
        assert channel.snr_db == 15.0
        assert channel.avg_noise_power is None
    
    def test_init_no_params(self):
        """Test initialization with no parameters."""
        with pytest.raises(ValueError):
            AWGNChannel()  # Should raise error when no parameters provided
    
    def test_forward_real_input(self):
        """Test forward pass with real input."""
        channel = AWGNChannel(avg_noise_power=0.1)
        x = torch.ones(100)
        y = channel(x)
        
        assert y.shape == x.shape
        assert not torch.is_complex(y)
        assert 0.08 <= torch.var(y - x).item() <= 0.12  # Noise variance should be close to 0.1
    
    def test_forward_complex_input(self):
        """Test forward pass with complex input."""
        channel = AWGNChannel(avg_noise_power=0.1)
        x = torch.complex(torch.ones(100), torch.zeros(100))
        y = channel(x)
        
        assert y.shape == x.shape
        assert torch.is_complex(y)
        # Noise should be split between real and imaginary parts
        assert 0.04 <= torch.var(y.real - x.real).item() <= 0.06
        assert 0.04 <= torch.var(y.imag - x.imag).item() <= 0.06
    
    def test_forward_with_snr(self):
        """Test forward pass with SNR specification."""
        channel = AWGNChannel(snr_db=10.0)
        x = torch.ones(100) * 2.0  # Signal power = 4.0
        y = channel(x)
        
        # Expected noise power = signal_power / 10^(snr_db/10) = 4.0 / 10 = 0.4
        assert 0.3 <= torch.var(y - x).item() <= 0.5
    
    def test_forward_with_pregenerated_noise(self):
        """Test forward pass with pre-generated noise."""
        channel = AWGNChannel(avg_noise_power=0.1)
        x = torch.ones(100)
        noise = torch.randn_like(x) * 0.2  # Custom noise
        y = channel(x, noise=noise)
        
        assert torch.allclose(y, x + noise)  # Should just add the provided noise


class TestLaplacianChannel:
    """Test suite for LaplacianChannel."""
    
    def test_init_parameters(self):
        """Test initialization with different parameters."""
        # Test with scale
        channel1 = LaplacianChannel(scale=0.5)
        assert channel1.scale.item() == 0.5
        assert channel1.avg_noise_power is None
        assert channel1.snr_db is None
        
        # Test with noise power
        channel2 = LaplacianChannel(avg_noise_power=0.1)
        assert channel2.avg_noise_power.item() == 0.1
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
    
    def test_get_laplacian_noise(self):
        """Test the _get_laplacian_noise method."""
        channel = LaplacianChannel(scale=1.0)
        noise = channel._get_laplacian_noise((1000,), "cpu")
        
        # Laplacian distribution should have mean close to 0
        assert -0.1 <= torch.mean(noise).item() <= 0.1
        
        # Variance of standard Laplacian is 2, but this is before scaling
        assert 1.8 <= torch.var(noise).item() <= 2.2
    
    def test_forward_real_input(self):
        """Test forward pass with real input and scale parameter."""
        channel = LaplacianChannel(scale=0.5)
        x = torch.ones(1000)
        y = channel(x)
        
        assert y.shape == x.shape
        assert not torch.is_complex(y)
        # Laplacian with scale 0.5 has variance = 2 * (0.5^2) = 0.5
        assert 0.4 <= torch.var(y - x).item() <= 0.6
    
    def test_forward_complex_input(self):
        """Test forward pass with complex input."""
        channel = LaplacianChannel(scale=0.5)
        x = torch.complex(torch.ones(1000), torch.zeros(1000))
        y = channel(x)
        
        assert y.shape == x.shape
        assert torch.is_complex(y)
        # Check that noise is added to both real and imaginary parts
        assert torch.var(y.real - x.real).item() > 0
        assert torch.var(y.imag - x.imag).item() > 0
    
    def test_forward_with_snr(self):
        """Test forward pass with SNR specification."""
        channel = LaplacianChannel(snr_db=10.0)
        x = torch.ones(1000) * 2.0  # Signal power = 4.0
        y = channel(x)
        
        # Expected noise power = signal_power / 10^(snr_db/10) = 4.0 / 10 = 0.4
        # Laplacian scale parameter would be sqrt(0.4/2) = sqrt(0.2)
        assert 0.3 <= torch.var(y - x).item() <= 0.5
    
    def test_forward_with_avg_noise_power(self):
        """Test forward pass with average noise power specification."""
        channel = LaplacianChannel(avg_noise_power=0.4)
        x = torch.ones(1000)
        y = channel(x)
        
        # Laplacian scale would be sqrt(0.4/2) = sqrt(0.2)
        assert 0.3 <= torch.var(y - x).item() <= 0.5


class TestPoissonChannel:
    """Test suite for PoissonChannel."""
    
    def test_init_parameters(self):
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
        x = torch.ones(100) * 0.5  # Input values
        y = channel(x)
        
        assert y.shape == x.shape
        assert not torch.is_complex(y)
        # Output should follow Poisson with rate = rate_factor * x
        # For Poisson, mean = variance = rate
        assert 4.0 <= torch.mean(y).item() <= 6.0  # Expected mean = 5.0
    
    def test_forward_with_normalization(self):
        """Test forward pass with normalization."""
        channel = PoissonChannel(rate_factor=10.0, normalize=True)
        x = torch.ones(1000) * 0.5  # Input values
        y = channel(x)
        
        # After normalization, output should be close to input on average
        assert 0.4 <= torch.mean(y).item() <= 0.6  # Expected mean ≈ 0.5
    
    def test_forward_negative_real_input(self):
        """Test forward pass with negative real input (should raise error)."""
        channel = PoissonChannel()
        x = torch.ones(10) * -1.0  # Negative values
        with pytest.raises(ValueError):
            channel(x)
    
    def test_forward_complex_input(self):
        """Test forward pass with complex input."""
        channel = PoissonChannel(rate_factor=10.0)
        x = torch.complex(torch.ones(100) * 0.5, torch.ones(100) * 0.5)
        y = channel(x)
        
        assert y.shape == x.shape
        assert torch.is_complex(y)
        # Phase should be preserved
        angles_x = torch.angle(x)
        angles_y = torch.angle(y)
        assert torch.allclose(angles_x, angles_y, atol=1e-5)
        
        # Magnitude should follow Poisson distribution
        magnitude_x = torch.abs(x)  # = sqrt(0.5^2 + 0.5^2) ≈ 0.707
        expected_rate = 10.0 * magnitude_x  # ≈ 7.07
        magnitude_y = torch.abs(y)
        assert 6.0 <= torch.mean(magnitude_y).item() <= 8.0


class TestPhaseNoiseChannel:
    """Test suite for PhaseNoiseChannel."""
    
    def test_init_parameters(self):
        """Test initialization with different parameters."""
        # Test with valid parameter
        channel = PhaseNoiseChannel(phase_noise_std=0.1)
        assert channel.phase_noise_std.item() == 0.1
        
        # Test with negative standard deviation (should raise error)
        with pytest.raises(ValueError):
            PhaseNoiseChannel(phase_noise_std=-0.1)
    
    def test_forward_real_input(self):
        """Test forward pass with real input."""
        channel = PhaseNoiseChannel(phase_noise_std=0.1)
        x = torch.ones(100)
        y = channel(x)
        
        assert y.shape == x.shape
        assert torch.is_complex(y)  # Output should be complex even if input is real
        # Magnitude should be approximately 1.0 since input is 1.0
        assert 0.95 <= torch.mean(torch.abs(y)).item() <= 1.05
    
    def test_forward_complex_input(self):
        """Test forward pass with complex input."""
        channel = PhaseNoiseChannel(phase_noise_std=0.1)
        # Create a complex tensor with constant magnitude
        magnitude = 2.0
        phase = torch.zeros(100)
        x = magnitude * torch.exp(1j * phase)
        y = channel(x)
        
        assert y.shape == x.shape
        assert torch.is_complex(y)
        # Magnitude should be preserved
        assert 1.95 <= torch.mean(torch.abs(y)).item() <= 2.05
        # Phase should be perturbed
        phases_y = torch.angle(y)
        assert torch.var(phases_y).item() > 0  # Phases should have variance
        # With phase_noise_std=0.1, the variance of phases should be close to 0.01
        assert 0.008 <= torch.var(phases_y).item() <= 0.012


class TestNonlinearChannel:
    """Test suite for NonlinearChannel."""
    
    def test_init_parameters(self):
        """Test initialization with different parameters."""
        # Test with simple nonlinearity, no noise
        def cubic(x): return x + 0.1 * x**3
        channel1 = NonlinearChannel(cubic)
        assert not channel1.add_noise
        assert channel1.complex_mode == "direct"
        
        # Test with noise parameters
        channel2 = NonlinearChannel(cubic, add_noise=True, avg_noise_power=0.1)
        assert channel2.add_noise
        assert channel2.avg_noise_power == 0.1
        assert channel2.snr_db is None
        
        # Test with SNR
        channel3 = NonlinearChannel(cubic, add_noise=True, snr_db=15.0)
        assert channel3.add_noise
        assert channel3.avg_noise_power is None
        assert channel3.snr_db == 15.0
        
        # Test with invalid parameters (adding noise but no power specified)
        with pytest.raises(ValueError):
            NonlinearChannel(cubic, add_noise=True)
            
        # Test with invalid complex mode
        with pytest.raises(ValueError):
            NonlinearChannel(cubic, complex_mode="invalid")
    
    def test_forward_real_input(self):
        """Test forward pass with real input."""
        def cubic(x): return x + 0.1 * x**3
        channel = NonlinearChannel(cubic)
        x = torch.linspace(-1, 1, 100)
        y = channel(x)
        
        assert y.shape == x.shape
        assert not torch.is_complex(y)
        # Expected output: y = x + 0.1*x^3
        expected = x + 0.1 * x**3
        assert torch.allclose(y, expected)
    
    def test_forward_complex_direct_mode(self):
        """Test forward pass with complex input in direct mode."""
        # A simple function that works on complex inputs
        def scale_complex(x): return 0.5 * x
        channel = NonlinearChannel(scale_complex, complex_mode="direct")
        x = torch.complex(torch.ones(10), torch.ones(10))
        y = channel(x)
        
        assert y.shape == x.shape
        assert torch.is_complex(y)
        # Expected output: y = 0.5 * x
        expected = 0.5 * x
        assert torch.allclose(y, expected)
    
    def test_forward_complex_cartesian_mode(self):
        """Test forward pass with complex input in cartesian mode."""
        def square(x): return x**2
        channel = NonlinearChannel(square, complex_mode="cartesian")
        x = torch.complex(torch.ones(10) * 2, torch.ones(10) * 3)
        y = channel(x)
        
        assert y.shape == x.shape
        assert torch.is_complex(y)
        # Expected output: y = real^2 + j*imag^2
        expected = torch.complex(x.real**2, x.imag**2)
        assert torch.allclose(y, expected)
    
    def test_forward_complex_polar_mode(self):
        """Test forward pass with complex input in polar mode."""
        def double_magnitude(x): return 2 * x
        channel = NonlinearChannel(double_magnitude, complex_mode="polar")
        # Create complex input with magnitude 1 and phase π/4
        x = torch.complex(torch.ones(10) / np.sqrt(2), torch.ones(10) / np.sqrt(2))
        y = channel(x)
        
        assert y.shape == x.shape
        assert torch.is_complex(y)
        # Magnitude should be doubled, phase preserved
        assert torch.allclose(torch.abs(y), torch.tensor(2.0))
        assert torch.allclose(torch.angle(y), torch.angle(x))
    
    def test_forward_with_noise(self):
        """Test forward pass with nonlinearity and noise."""
        def identity(x): return x  # Simple identity function
        channel = NonlinearChannel(identity, add_noise=True, avg_noise_power=0.1)
        x = torch.ones(100)
        y = channel(x)
        
        assert y.shape == x.shape
        # Output should be input plus noise
        assert 0.08 <= torch.var(y - x).item() <= 0.12  # Noise variance should be close to 0.1