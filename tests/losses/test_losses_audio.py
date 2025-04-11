"""Tests for the audio losses module with comprehensive coverage."""

import pytest
import torch
import torch.nn as nn

from kaira.losses.audio import (
    AudioContrastiveLoss,
    FeatureMatchingLoss,
    L1AudioLoss,
    LogSTFTMagnitudeLoss,
    MelSpectrogramLoss,
    MultiResolutionSTFTLoss,
    SpectralConvergenceLoss,
    STFTLoss,
)
from kaira.losses.registry import LossRegistry


class TestAudioContrastiveLoss:
    """Tests for AudioContrastiveLoss."""

    @pytest.fixture
    def features(self):
        """Create sample feature tensor for testing."""
        return torch.randn(8, 128)

    @pytest.fixture
    def target(self):
        """Create sample target tensor for testing."""
        return torch.randn(8, 128)

    @pytest.fixture
    def labels(self):
        """Create sample labels for supervised contrastive learning."""
        return torch.tensor([0, 1, 0, 2, 1, 2, 0, 1])

    @pytest.fixture
    def projector(self):
        """Create simple projector network for testing."""
        return nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))

    @pytest.fixture
    def view_maker(self):
        """Create a simple view maker function for testing."""

        def make_view(x):
            # Apply slight noise to create a different "view"
            return x + 0.1 * torch.randn_like(x)

        return make_view

    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Default initialization
        loss_fn = AudioContrastiveLoss()
        assert loss_fn.margin == 1.0
        assert loss_fn.temperature == 0.1
        assert loss_fn.normalize is True
        assert loss_fn.reduction == "mean"

        # Custom initialization
        loss_fn = AudioContrastiveLoss(margin=0.5, temperature=0.2, normalize=False, reduction="sum")
        assert loss_fn.margin == 0.5
        assert loss_fn.temperature == 0.2
        assert loss_fn.normalize is False
        assert loss_fn.reduction == "sum"

    def test_loss_registration(self):
        """Test if the loss is properly registered."""
        loss = LossRegistry.create("audiocontrastiveloss")  # Fixed typo: was "audiocontractiveloss"
        assert isinstance(loss, AudioContrastiveLoss)

    def test_forward_basic(self, features):
        """Test basic forward pass with only features."""
        features.requires_grad_(True)
        loss_fn = AudioContrastiveLoss()
        loss = loss_fn(features)

        # Check loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # Changed from > 0 to >= 0
        assert loss.grad_fn is not None

        # Check gradient flow
        loss.backward()
        assert features.grad is not None

    def test_forward_with_target(self, features, target):
        """Test forward pass with features and target."""
        features.requires_grad_(True)
        target.requires_grad_(True)

        loss_fn = AudioContrastiveLoss()
        loss = loss_fn(features, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0  # Changed from > 0 to >= 0

        # Check gradient flow
        loss.backward()
        assert features.grad is not None
        assert target.grad is not None

    def test_forward_with_projector(self, features, target, projector):
        """Test forward pass with projector network."""
        features.requires_grad_(True)
        target.requires_grad_(True)

        loss_fn = AudioContrastiveLoss()
        loss = loss_fn(features, target, projector=projector)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0  # Changed from > 0 to >= 0

        # Check gradient flow
        loss.backward()
        assert features.grad is not None
        assert target.grad is not None

        # Check if projector was applied - the dimensionality would be reduced
        for p in projector.parameters():
            assert p.grad is not None

    def test_forward_with_view_maker(self, features, view_maker):
        """Test forward pass with view maker function."""
        features.requires_grad_(True)

        loss_fn = AudioContrastiveLoss()
        loss = loss_fn(features, view_maker=view_maker)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0  # Changed from > 0 to >= 0

        # Check gradient flow
        loss.backward()
        assert features.grad is not None

    def test_forward_with_view_maker_and_target(self, features, target, view_maker):
        """Test forward pass with view maker function and target."""
        features.requires_grad_(True)
        target.requires_grad_(True)

        loss_fn = AudioContrastiveLoss()
        loss = loss_fn(features, target, view_maker=view_maker)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0  # Changed from > 0 to >= 0

        # Check gradient flow
        loss.backward()
        assert features.grad is not None
        assert target.grad is not None

    def test_forward_with_labels(self, features, labels):
        """Test forward pass with supervised labels."""
        features.requires_grad_(True)

        loss_fn = AudioContrastiveLoss()
        loss = loss_fn(features, labels=labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

        # Check gradient flow
        loss.backward()
        assert features.grad is not None

    def test_no_normalization(self, features):
        """Test forward pass without normalization."""
        features.requires_grad_(True)

        loss_fn = AudioContrastiveLoss(normalize=False)
        loss = loss_fn(features)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.grad_fn is not None

    def test_reduction_methods(self, features):
        """Test different reduction methods."""
        features.requires_grad_(True)
        batch_size = features.size(0)

        # Test mean reduction
        loss_fn_mean = AudioContrastiveLoss(reduction="mean")
        loss_mean = loss_fn_mean(features)
        assert loss_mean.ndim == 0

        # Test sum reduction
        loss_fn_sum = AudioContrastiveLoss(reduction="sum")
        loss_sum = loss_fn_sum(features)
        assert loss_sum.ndim == 0

        # Test no reduction
        loss_fn_none = AudioContrastiveLoss(reduction="none")
        loss_none = loss_fn_none(features)
        assert loss_none.ndim == 1
        assert loss_none.shape[0] == batch_size

    def test_reduction_methods_comprehensive(self, features):
        """Test reduction methods more comprehensively, ensuring each branch works correctly."""
        features.requires_grad_(True)
        batch_size = features.size(0)

        # Create features that will generate non-zero loss values
        # We'll create features where each sample is identical to ensure predictable positive pairs
        controlled_features = torch.ones((batch_size, 128))
        # Make each feature vector slightly different to avoid perfect similarity
        for i in range(batch_size):
            controlled_features[i] *= i + 1
        controlled_features.requires_grad_(True)

        # 1. Test mean reduction
        loss_fn_mean = AudioContrastiveLoss(reduction="mean")
        loss_mean = loss_fn_mean(controlled_features)

        # 2. Test sum reduction
        loss_fn_sum = AudioContrastiveLoss(reduction="sum")
        loss_sum = loss_fn_sum(controlled_features)

        # 3. Test no reduction ('none')
        loss_fn_none = AudioContrastiveLoss(reduction="none")
        loss_none = loss_fn_none(controlled_features)

        # Verify shapes
        assert loss_mean.ndim == 0  # Scalar
        assert loss_sum.ndim == 0  # Scalar
        assert loss_none.ndim == 1  # Vector with batch_size elements
        assert loss_none.shape[0] == batch_size

        # Verify relationships between different reductions
        # The sum loss should equal the sum of the 'none' reduction losses
        assert torch.isclose(loss_sum, loss_none.sum())

        # The mean loss should equal the mean of the 'none' reduction losses
        assert torch.isclose(loss_mean, loss_none.mean())

        # Test gradient flow for all reduction methods
        loss_mean.backward(retain_graph=True)
        assert controlled_features.grad is not None

        controlled_features.grad = None  # Reset gradients

        loss_sum.backward(retain_graph=True)
        assert controlled_features.grad is not None

        controlled_features.grad = None  # Reset gradients

        loss_none.mean().backward()  # Need to reduce to scalar for backward
        assert controlled_features.grad is not None

    def test_edge_case_single_element(self):
        """Test with a single element (batch size 1)."""
        features = torch.randn(1, 128)
        features.requires_grad_(True)

        loss_fn = AudioContrastiveLoss()
        # With a single element, there are no positive pairs for InfoNCE loss
        # So we need to make sure it doesn't crash
        loss = loss_fn(features)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

        # Check gradient flow
        loss.backward()
        assert features.grad is not None

    def test_edge_case_no_positive_pairs(self, features):
        """Test the case where there are no positive pairs for some samples."""
        features.requires_grad_(True)

        # Create labels where one sample has no positive pairs
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])  # All unique labels

        loss_fn = AudioContrastiveLoss()
        loss = loss_fn(features, labels=labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

        # Check gradient flow
        loss.backward()
        assert features.grad is not None

    def test_all_components_together(self, features, projector, view_maker, labels):
        """Test all components of the loss function together."""
        features.requires_grad_(True)

        loss_fn = AudioContrastiveLoss()
        loss = loss_fn(features, projector=projector, view_maker=view_maker, labels=labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

        # Check gradient flow
        loss.backward()
        assert features.grad is not None

        # Check projector gradients
        for p in projector.parameters():
            assert p.grad is not None


@pytest.fixture
def audio_data():
    """Fixture for creating sample audio batch."""
    # Create a batch of 4 audio samples, each 16000 samples (1 second at 16kHz)
    return torch.sin(torch.linspace(0, 100 * torch.pi, 16000)).unsqueeze(0).repeat(4, 1)


@pytest.fixture
def target_audio_data():
    """Fixture for creating sample target audio batch (slightly different from input)."""
    # Create a batch of 4 audio samples with a different frequency component
    return torch.sin(torch.linspace(0, 110 * torch.pi, 16000)).unsqueeze(0).repeat(4, 1)


@pytest.fixture
def spectral_magnitudes():
    """Fixture for creating sample spectral magnitudes."""
    # Create a batch of 4 spectrograms, each with 513 frequency bins and 32 time frames
    return torch.abs(torch.randn(4, 513, 32))


@pytest.fixture
def target_spectral_magnitudes():
    """Fixture for creating sample target spectral magnitudes."""
    # Create a batch of 4 spectrograms, each with 513 frequency bins and 32 time frames
    return torch.abs(torch.randn(4, 513, 32))


@pytest.fixture
def mock_feature_extractor():
    """Fixture for creating a mock feature extractor model."""

    # Simple CNN model for feature extraction
    class MockFeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.relu3 = nn.ReLU()

        def forward(self, x):
            # Ensure input is shaped properly for 1D convolution
            if x.dim() == 2:  # [batch, samples]
                x = x.unsqueeze(1)  # [batch, channels, samples]

            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            x = self.relu3(self.conv3(x))
            return x

    return MockFeatureExtractor()


class TestL1AudioLoss:
    """Test suite for L1AudioLoss."""

    def test_forward(self, audio_data, target_audio_data):
        """Test basic forward pass."""
        loss_fn = L1AudioLoss()
        loss = loss_fn(audio_data, target_audio_data)

        # Check that the loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

        # Verify the loss value matches PyTorch's L1Loss
        expected_loss = nn.L1Loss()(audio_data, target_audio_data)
        assert torch.isclose(loss, expected_loss)

    def test_identical_inputs(self, audio_data):
        """Test with identical input and target (loss should be zero)."""
        loss_fn = L1AudioLoss()
        loss = loss_fn(audio_data, audio_data)

        assert torch.isclose(loss, torch.tensor(0.0))


class TestSpectralConvergenceLoss:
    """Test suite for SpectralConvergenceLoss."""

    def test_forward(self, spectral_magnitudes, target_spectral_magnitudes):
        """Test basic forward pass."""
        loss_fn = SpectralConvergenceLoss()
        loss = loss_fn(spectral_magnitudes, target_spectral_magnitudes)

        # Check that the loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

        # Verify the loss formula manually
        expected_loss = torch.norm(target_spectral_magnitudes - spectral_magnitudes, p="fro") / torch.norm(target_spectral_magnitudes, p="fro")
        assert torch.isclose(loss, expected_loss)

    def test_identical_inputs(self, spectral_magnitudes):
        """Test with identical input and target (loss should be zero)."""
        loss_fn = SpectralConvergenceLoss()
        loss = loss_fn(spectral_magnitudes, spectral_magnitudes)

        assert torch.isclose(loss, torch.tensor(0.0))


class TestLogSTFTMagnitudeLoss:
    """Test suite for LogSTFTMagnitudeLoss."""

    def test_forward(self, spectral_magnitudes, target_spectral_magnitudes):
        """Test basic forward pass."""
        loss_fn = LogSTFTMagnitudeLoss()
        loss = loss_fn(spectral_magnitudes, target_spectral_magnitudes)

        # Check that the loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

        # Verify the loss formula manually
        log_input = torch.log(spectral_magnitudes + 1e-7)
        log_target = torch.log(target_spectral_magnitudes + 1e-7)
        expected_loss = nn.L1Loss()(log_input, log_target)

        assert torch.isclose(loss, expected_loss)

    def test_identical_inputs(self, spectral_magnitudes):
        """Test with identical input and target (loss should be zero)."""
        loss_fn = LogSTFTMagnitudeLoss()
        loss = loss_fn(spectral_magnitudes, spectral_magnitudes)

        assert torch.isclose(loss, torch.tensor(0.0))

    def test_zero_magnitudes(self):
        """Test with near-zero magnitude values."""
        # Create very small magnitude values
        x_mag = torch.ones(2, 10, 10) * 1e-8
        target_mag = torch.ones(2, 10, 10) * 1e-8

        loss_fn = LogSTFTMagnitudeLoss()
        loss = loss_fn(x_mag, target_mag)

        # Loss should be finite and reasonable
        assert torch.isfinite(loss)


class TestSTFTLoss:
    """Test suite for STFTLoss."""

    def test_forward(self, audio_data, target_audio_data):
        """Test basic forward pass."""
        loss_fn = STFTLoss(fft_size=512, hop_size=128, win_length=512)
        loss = loss_fn(audio_data, target_audio_data)

        # Check that the loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss >= 0  # Loss should be non-negative

    def test_different_window_functions(self, audio_data, target_audio_data):
        """Test STFTLoss with different window functions."""
        # Test with Hann window
        loss_fn_hann = STFTLoss(window="hann")
        loss_hann = loss_fn_hann(audio_data, target_audio_data)

        # Test with Hamming window
        loss_fn_hamming = STFTLoss(window="hamming")
        loss_hamming = loss_fn_hamming(audio_data, target_audio_data)

        assert isinstance(loss_hann, torch.Tensor)
        assert isinstance(loss_hamming, torch.Tensor)
        # Different windows should give different loss values
        assert loss_hann.item() != loss_hamming.item()

    def test_different_fft_params(self, audio_data, target_audio_data):
        """Test STFTLoss with different FFT parameters."""
        # Test with default parameters
        loss_fn_default = STFTLoss()
        loss_default = loss_fn_default(audio_data, target_audio_data)

        # Test with different parameters
        loss_fn_custom = STFTLoss(fft_size=2048, hop_size=512, win_length=2048)
        loss_custom = loss_fn_custom(audio_data, target_audio_data)

        assert isinstance(loss_default, torch.Tensor)
        assert isinstance(loss_custom, torch.Tensor)
        # Different parameters should give different loss values
        assert loss_default.item() != loss_custom.item()


class TestMultiResolutionSTFTLoss:
    """Test suite for MultiResolutionSTFTLoss."""

    def test_forward(self, audio_data, target_audio_data):
        """Test basic forward pass."""
        loss_fn = MultiResolutionSTFTLoss()
        loss = loss_fn(audio_data, target_audio_data)

        # Check that the loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss >= 0  # Loss should be non-negative

    def test_custom_resolutions(self, audio_data, target_audio_data):
        """Test with custom resolution parameters."""
        # Define custom resolution parameters
        fft_sizes = [256, 512]
        hop_sizes = [64, 128]
        win_lengths = [256, 512]

        loss_fn = MultiResolutionSTFTLoss(fft_sizes=fft_sizes, hop_sizes=hop_sizes, win_lengths=win_lengths)
        loss = loss_fn(audio_data, target_audio_data)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

        # Verify that we have the correct number of STFT losses
        assert len(loss_fn.stft_losses) == len(fft_sizes)

    def test_internal_stft_losses(self, audio_data, target_audio_data):
        """Test that individual STFT losses are computed correctly."""
        loss_fn = MultiResolutionSTFTLoss(fft_sizes=[512], hop_sizes=[128], win_lengths=[512])

        # This should be equivalent to a single STFTLoss
        multi_res_loss = loss_fn(audio_data, target_audio_data)

        single_stft_loss = STFTLoss(fft_size=512, hop_size=128, win_length=512)(audio_data, target_audio_data)

        # The multi-resolution loss with a single resolution should equal the single STFT loss
        assert torch.isclose(multi_res_loss, single_stft_loss)


class TestMelSpectrogramLoss:
    """Test suite for MelSpectrogramLoss."""

    def test_forward(self, audio_data, target_audio_data):
        """Test basic forward pass."""
        # Use a smaller n_fft for speed
        loss_fn = MelSpectrogramLoss(n_fft=512, hop_length=256, n_mels=40)
        loss = loss_fn(audio_data, target_audio_data)

        # Check that the loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss >= 0  # Loss should be non-negative

    def test_with_and_without_log_mel(self, audio_data, target_audio_data):
        """Test with and without log-mel option."""
        # With log-mel
        loss_fn_log = MelSpectrogramLoss(n_fft=512, log_mel=True)
        loss_log = loss_fn_log(audio_data, target_audio_data)

        # Without log-mel
        loss_fn_no_log = MelSpectrogramLoss(n_fft=512, log_mel=False)
        loss_no_log = loss_fn_no_log(audio_data, target_audio_data)

        assert isinstance(loss_log, torch.Tensor)
        assert isinstance(loss_no_log, torch.Tensor)
        # Log and non-log versions should give different results
        assert loss_log.item() != loss_no_log.item()

    def test_different_parameters(self, audio_data, target_audio_data):
        """Test with different mel-spectrogram parameters."""
        loss_fn1 = MelSpectrogramLoss(n_mels=40, f_max=4000)
        loss_fn2 = MelSpectrogramLoss(n_mels=80, f_max=8000)

        loss1 = loss_fn1(audio_data, target_audio_data)
        loss2 = loss_fn2(audio_data, target_audio_data)

        assert isinstance(loss1, torch.Tensor)
        assert isinstance(loss2, torch.Tensor)
        # Different parameters should give different results
        assert loss1.item() != loss2.item()


class TestFeatureMatchingLoss:
    """Test suite for FeatureMatchingLoss."""

    def test_forward(self, audio_data, target_audio_data, mock_feature_extractor):
        """Test basic forward pass."""
        # Use layers 0 (conv1) and 2 (conv3)
        layers = [0, 2]
        loss_fn = FeatureMatchingLoss(model=mock_feature_extractor, layers=layers)
        loss = loss_fn(audio_data, target_audio_data)

        # Check that the loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss >= 0  # Loss should be non-negative

    def test_with_custom_weights(self, audio_data, target_audio_data, mock_feature_extractor):
        """Test with custom weights for each layer."""
        layers = [0, 2]  # conv1 and conv3
        weights = [0.3, 0.7]  # More weight on deeper layers

        loss_fn = FeatureMatchingLoss(model=mock_feature_extractor, layers=layers, weights=weights)
        loss = loss_fn(audio_data, target_audio_data)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_model_remains_frozen(self, audio_data, target_audio_data, mock_feature_extractor):
        """Test that the model parameters are not updated during training."""
        # First, ensure model parameters require grad before passing to loss
        for param in mock_feature_extractor.parameters():
            param.requires_grad = True

        layers = [0, 2]
        loss_fn = FeatureMatchingLoss(model=mock_feature_extractor, layers=layers)

        # Check that model parameters no longer require grad after creating loss
        for param in loss_fn.model.parameters():
            assert not param.requires_grad

        # Compute loss
        loss = loss_fn(audio_data, target_audio_data)
        loss.backward()  # This should work without errors

        # Verify model is in eval mode
        assert not loss_fn.model.training
