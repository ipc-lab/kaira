# tests/test_metrics.py
import pytest
import torch

from kaira.metrics import (
    LearnedPerceptualImagePatchSimilarity,
    MultiScaleSSIM,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    BitErrorRate,
    BlockErrorRate,
    FrameErrorRate,
    SymbolErrorRate,
    SignalToNoiseRatio,
)


@pytest.fixture
def sample_preds():
    """Fixture for creating sample predictions tensor."""
    # Increased size to accommodate multi-scale operations (at least 256x256)
    return torch.randn(1, 3, 256, 256)


@pytest.fixture
def sample_targets():
    """Fixture for creating sample targets tensor."""
    # Increased size to accommodate multi-scale operations (at least 256x256)
    return torch.randn(1, 3, 256, 256)


@pytest.fixture
def signal_data():
    """Fixture for creating sample signal data."""
    torch.manual_seed(42)
    signal = torch.randn(1, 1000)  # Original signal
    noise = 0.1 * torch.randn(1, 1000)  # Noise
    noisy_signal = signal + noise  # Noisy signal
    return signal, noisy_signal

@pytest.fixture
def binary_data():
    """Fixture for creating sample binary data."""
    torch.manual_seed(42)
    true_bits = torch.randint(0, 2, (1, 1000))
    error_mask = torch.rand(1, 1000) < 0.05  # 5% error rate
    received_bits = torch.logical_xor(true_bits, error_mask).int()
    return true_bits, received_bits


def test_multiscale_ssim_initialization():
    """Test MultiScaleSSIM initialization."""
    msssim = MultiScaleSSIM()
    assert isinstance(msssim, MultiScaleSSIM)


def test_multiscale_ssim_update(sample_preds, sample_targets):
    """Test MultiScaleSSIM update method."""
    msssim = MultiScaleSSIM()
    msssim.update(sample_preds, sample_targets)
    assert msssim.sum_values.shape == torch.Size([])  # Use sum_values instead of sum
    assert msssim.count.shape == torch.Size([])  # Check count instead of total


def test_multiscale_ssim_forward(sample_preds, sample_targets):
    """Test MultiScaleSSIM forward pass."""
    msssim = MultiScaleSSIM()
    msssim.update(sample_preds, sample_targets)
    mean, std = msssim.compute()  # Unpack the tuple correctly
    assert isinstance(mean, torch.Tensor)
    assert isinstance(std, torch.Tensor)


def test_lpips_initialization():
    """Test LearnedPerceptualImagePatchSimilarity initialization."""
    lpips = LearnedPerceptualImagePatchSimilarity()
    assert isinstance(lpips, LearnedPerceptualImagePatchSimilarity)


def test_lpips_update(sample_preds, sample_targets):
    """Test LearnedPerceptualImagePatchSimilarity update method."""
    lpips = LearnedPerceptualImagePatchSimilarity()
    lpips.update(sample_preds, sample_targets)
    assert lpips.sum_scores.shape == torch.Size([])
    assert lpips.total.shape == torch.Size([])


def test_lpips_compute(sample_preds, sample_targets):
    """Test LearnedPerceptualImagePatchSimilarity compute method."""
    lpips = LearnedPerceptualImagePatchSimilarity()
    lpips.update(sample_preds, sample_targets)
    mean, std = lpips.compute()
    assert isinstance(mean, torch.Tensor)
    assert isinstance(std, torch.Tensor)


def test_psnr_initialization():
    """Test PeakSignalNoiseRatio initialization."""
    psnr = PeakSignalNoiseRatio()
    assert isinstance(psnr, PeakSignalNoiseRatio)


def test_psnr_compute(sample_preds, sample_targets):
    """Test PeakSignalNoiseRatio compute method."""
    psnr = PeakSignalNoiseRatio()
    # Direct computation instead of using update
    psnr_value = psnr(sample_preds, sample_targets)
    assert isinstance(psnr_value, torch.Tensor)
    assert psnr_value.ndim == 0  # Scalar output
    assert not torch.isnan(psnr_value)  # Should not be NaN


def test_ssim_initialization():
    """Test StructuralSimilarityIndexMeasure initialization."""
    ssim = StructuralSimilarityIndexMeasure()
    assert isinstance(ssim, StructuralSimilarityIndexMeasure)


def test_ssim_compute(sample_preds, sample_targets):
    """Test StructuralSimilarityIndexMeasure compute method."""
    ssim = StructuralSimilarityIndexMeasure()
    # Direct computation instead of using update
    ssim_value = ssim(sample_preds, sample_targets)
    assert isinstance(ssim_value, torch.Tensor)
    assert ssim_value.ndim == 0  # Scalar output
    assert 0 <= ssim_value <= 1  # SSIM is between 0 and 1


def test_snr_initialization():
    """Test SignalToNoiseRatio initialization."""
    snr = SignalToNoiseRatio()
    assert isinstance(snr, SignalToNoiseRatio)

def test_snr_computation(signal_data):
    """Test SNR computation."""
    signal, noisy_signal = signal_data
    snr = SignalToNoiseRatio()
    snr_value = snr(noisy_signal, signal)
    assert isinstance(snr_value, torch.Tensor)
    assert snr_value.ndim == 0  # Scalar output
    assert snr_value > 0  # SNR should be positive

@pytest.mark.parametrize("snr_db", [-10, 0, 10, 20])
def test_snr_db_values(signal_data, snr_db):
    """Test SNR computation with different SNR values."""
    signal, _ = signal_data
    
    # Create noisy signal with specific SNR using correct power calculation
    signal_power = torch.mean(signal**2).item()
    noise_power = signal_power / (10**(snr_db/10))  # Calculate required noise power for desired SNR
    
    # Generate noise with the exact power needed
    noise = torch.randn_like(signal)
    noise_scale = torch.sqrt(torch.tensor(noise_power) / torch.mean(noise**2))
    scaled_noise = noise * noise_scale
    
    # Create noisy signal
    noisy_signal = signal + scaled_noise
    
    # Calculate SNR
    snr = SignalToNoiseRatio()
    snr_value = snr(signal, noisy_signal)
    
    # Check the calculated SNR is close to the expected value
    assert abs(snr_value.item() - snr_db) < 1.0  # Allow for some numerical error

def test_snr_perfect_signal():
    """Test SNR computation with no noise."""
    signal = torch.randn(1, 100)
    snr = SignalToNoiseRatio()
    snr_value = snr(signal, signal)
    assert torch.isinf(snr_value)

def test_snr_batch_computation():
    """Test SNR computation with batched inputs."""
    batch_size = 3
    signal = torch.randn(batch_size, 100)
    noise = 0.1 * torch.randn_like(signal)
    noisy_signal = signal + noise
    
    snr = SignalToNoiseRatio()
    snr_values = snr(noisy_signal, signal)
    assert snr_values.shape == (batch_size,)
    assert torch.all(snr_values > 0)

def test_snr_complex_signal():
    """Test SNR computation with complex signals."""
    signal = torch.complex(torch.randn(1, 100), torch.randn(1, 100))
    noise = 0.1 * torch.complex(torch.randn_like(signal.real), torch.randn_like(signal.imag))
    noisy_signal = signal + noise
    
    snr = SignalToNoiseRatio()
    snr_value = snr(noisy_signal, signal)
    assert isinstance(snr_value, torch.Tensor)
    assert snr_value > 0

def test_ber_initialization():
    """Test BitErrorRate initialization."""
    ber = BitErrorRate()
    assert isinstance(ber, BitErrorRate)

def test_ber_computation(binary_data):
    """Test BER computation."""
    true_bits, received_bits = binary_data
    ber = BitErrorRate()
    ber_value = ber(received_bits, true_bits)
    assert isinstance(ber_value, torch.Tensor)
    assert ber_value.ndim == 0  # Scalar output
    assert 0 <= ber_value <= 1  # BER should be between 0 and 1

def test_ber_zero_errors(binary_data):
    """Test BER computation with zero errors."""
    true_bits, _ = binary_data
    ber = BitErrorRate()
    ber_value = ber(true_bits, true_bits)  # Compare with itself
    assert ber_value == 0.0

def test_ber_all_errors():
    """Test BER computation with all errors."""
    bits = torch.zeros((1, 100))
    inverted_bits = torch.ones((1, 100))
    ber = BitErrorRate()
    ber_value = ber(inverted_bits, bits)
    assert ber_value == 1.0

def test_ber_threshold():
    """Test BER with different decision thresholds."""
    true_bits = torch.zeros((1, 100))
    soft_decisions = torch.linspace(-1, 1, 100).reshape(1, -1)
    ber = BitErrorRate(threshold=0.0)  # Middle threshold
    ber_value = ber(soft_decisions, true_bits)
    assert 0 <= ber_value <= 1

@pytest.mark.parametrize("error_rate", [0.0, 0.1, 0.5, 1.0])
def test_ber_specific_error_rates(error_rate):
    """Test BER computation with specific error rates."""
    n_bits = 1000
    true_bits = torch.zeros((1, n_bits))
    errors = torch.rand(1, n_bits) < error_rate
    received_bits = torch.logical_xor(true_bits, errors).int()
    
    ber = BitErrorRate()
    ber_value = ber(received_bits, true_bits)
    assert abs(ber_value.item() - error_rate) < 0.05  # Allow for statistical variation

@pytest.mark.parametrize("block_size", [10, 50, 100])
def test_bler_computation(binary_data, block_size):
    """Test BLER computation with different block sizes."""
    true_bits, received_bits = binary_data
    n_blocks = true_bits.size(1) // block_size
    usable_bits = n_blocks * block_size
    
    # Reshape into blocks
    true_blocks = true_bits[:, :usable_bits].reshape(1, -1, block_size)
    received_blocks = received_bits[:, :usable_bits].reshape(1, -1, block_size)
    
    bler = BlockErrorRate()
    # Direct computation instead of update+compute
    bler_value = bler(received_blocks, true_blocks)
    assert isinstance(bler_value, torch.Tensor)
    assert bler_value.ndim == 0  # Scalar output
    assert 0 <= bler_value <= 1  # BLER should be between 0 and 1
    
    # Test the update+compute path separately
    bler.reset()
    bler.update(received_blocks, true_blocks)
    bler_mean, bler_std = bler.compute()  # Now properly unpack the tuple
    assert isinstance(bler_mean, torch.Tensor)
    assert isinstance(bler_std, torch.Tensor)

def test_block_error_edge_cases():
    """Test BLER computation with edge cases."""
    bler = BlockErrorRate()
    
    # Test perfect transmission
    perfect_blocks = torch.zeros((1, 10, 8))
    bler_value = bler(perfect_blocks, perfect_blocks)
    assert bler_value == 0.0
    
    # Test completely corrupted transmission
    corrupted_blocks = torch.ones((1, 10, 8))
    clean_blocks = torch.zeros((1, 10, 8))
    bler_value = bler(corrupted_blocks, clean_blocks)
    assert bler_value == 1.0

@pytest.mark.parametrize("block_size,error_pattern", [
    (10, [0]),          # Error in first block
    (20, [1, 3]),       # Errors in middle blocks
    (50, [-1]),         # Error in last block
    (100, [0, -1])      # Errors in first and last blocks
])
def test_bler_specific_patterns(block_size, error_pattern):
    """Test BLER computation with specific error patterns."""
    n_blocks = 10
    n_bits = block_size * n_blocks
    true_bits = torch.zeros((1, n_bits))
    received_bits = true_bits.clone()
    
    # Introduce errors in specific blocks
    for block_idx in error_pattern:
        start_idx = block_idx * block_size
        received_bits[0, start_idx] = 1  # Introduce at least one error in the block
    
    blocks = true_bits.reshape(1, n_blocks, block_size)
    received_blocks = received_bits.reshape(1, n_blocks, block_size)
    
    bler = BlockErrorRate()
    bler_value = bler(received_blocks, blocks)
    expected_bler = len(error_pattern) / n_blocks
    assert abs(bler_value.item() - expected_bler) < 1e-6

@pytest.mark.parametrize("frame_size", [100, 200])
def test_fer_computation(binary_data, frame_size):
    """Test FER computation with different frame sizes."""
    true_bits, received_bits = binary_data
    n_frames = true_bits.size(1) // frame_size
    usable_bits = n_frames * frame_size
    
    # Reshape into frames
    true_frames = true_bits[:, :usable_bits].reshape(1, -1, frame_size)
    received_frames = received_bits[:, :usable_bits].reshape(1, -1, frame_size)
    
    fer = FrameErrorRate()
    # Direct computation
    fer_value = fer(received_frames, true_frames)
    assert isinstance(fer_value, torch.Tensor)
    assert fer_value.ndim == 0  # Scalar output
    assert 0 <= fer_value <= 1  # FER should be between 0 and 1
    
    # Test update + compute path
    fer.reset()
    fer.update(received_frames, true_frames)
    fer_mean, fer_std = fer.compute()  # Properly unpack the tuple
    assert isinstance(fer_mean, torch.Tensor)
    assert isinstance(fer_std, torch.Tensor)

@pytest.mark.parametrize("bits_per_symbol", [2, 4, 6])  # Testing for QPSK, 16-QAM, 64-QAM
def test_ser_computation(binary_data, bits_per_symbol):
    """Test SER computation with different modulation orders."""
    true_bits, received_bits = binary_data
    n_symbols = true_bits.size(1) // bits_per_symbol
    usable_bits = n_symbols * bits_per_symbol
    
    # Reshape into symbols
    true_symbols = true_bits[:, :usable_bits].reshape(1, -1, bits_per_symbol)
    received_symbols = received_bits[:, :usable_bits].reshape(1, -1, bits_per_symbol)
    
    ser = SymbolErrorRate()
    # Direct computation
    ser_value = ser(received_symbols, true_symbols)
    assert isinstance(ser_value, torch.Tensor)
    assert ser_value.ndim == 0  # Scalar output
    assert 0 <= ser_value <= 1  # SER should be between 0 and 1
    
    # Test update + compute path
    ser.reset()
    ser.update(received_symbols, true_symbols)
    ser_mean, ser_std = ser.compute()  # Properly unpack the tuple
    assert isinstance(ser_mean, torch.Tensor)
    assert isinstance(ser_std, torch.Tensor)

@pytest.mark.parametrize("error_positions", [0, -1, "middle"])
def test_ser_single_error(error_positions):
    """Test SER computation with single error in different positions."""
    ser = SymbolErrorRate()
    true_symbols = torch.zeros((1, 10, 4))  # 10 symbols, 4 bits each
    received_symbols = true_symbols.clone()
    
    if error_positions == "middle":
        error_pos = 5
    else:
        error_pos = error_positions
    
    received_symbols[0, error_pos, 0] = 1  # Introduce single error
    ser_value = ser(received_symbols, true_symbols)
    assert ser_value == 0.1  # One error in 10 symbols

@pytest.mark.parametrize("bits_per_symbol,error_positions", [
    (2, [0]),     # QPSK with error in first symbol
    (4, [1, 2]),  # 16-QAM with errors in middle symbols
    (6, [-1])     # 64-QAM with error in last symbol
])
def test_ser_specific_positions(bits_per_symbol, error_positions):
    """Test SER computation with errors in specific positions."""
    n_symbols = 10
    true_symbols = torch.zeros((1, n_symbols, bits_per_symbol))
    received_symbols = true_symbols.clone()
    
    # Introduce errors at specific positions
    for pos in error_positions:
        # Flip the bits to create errors (1s instead of 0s)
        received_symbols[0, pos] = torch.ones_like(received_symbols[0, pos])
    
    ser = SymbolErrorRate()
    ser_value = ser(received_symbols, true_symbols)
    expected_ser = len(error_positions) / n_symbols
    assert abs(ser_value.item() - expected_ser) < 1e-6

@pytest.mark.parametrize("kernel_size", [7, 11, 15])
def test_multiscale_ssim_different_kernel_sizes(sample_preds, sample_targets, kernel_size):
    """Test MultiScaleSSIM with different kernel sizes."""
    msssim = MultiScaleSSIM(kernel_size=kernel_size)
    msssim.update(sample_preds, sample_targets)
    mean, std = msssim.compute()  # Unpack the tuple correctly
    assert isinstance(mean, torch.Tensor)
    assert isinstance(std, torch.Tensor)


@pytest.mark.parametrize("net_type", ["vgg", "alex", "squeeze"])
def test_lpips_different_net_types(sample_preds, sample_targets, net_type):
    """Test LearnedPerceptualImagePatchSimilarity with different net_type values."""
    lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type)
    lpips.update(sample_preds, sample_targets)
    mean, std = lpips.compute()
    assert isinstance(mean, torch.Tensor)
    assert isinstance(std, torch.Tensor)
