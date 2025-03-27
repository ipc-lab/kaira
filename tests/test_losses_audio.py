import pytest
import torch

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


@pytest.fixture
def sample_audio():
    """Fixture for creating sample audio tensors."""
    return torch.randn(1, 22050)  # 1 second of audio at 22.05 kHz


@pytest.fixture
def target_audio():
    """Fixture for creating target audio tensors."""
    return torch.randn(1, 22050)  # 1 second of audio at 22.05 kHz


def test_l1_audio_loss_forward(sample_audio, target_audio):
    """Test L1AudioLoss forward pass."""
    l1_audio_loss = L1AudioLoss()
    loss = l1_audio_loss(sample_audio, target_audio)
    assert isinstance(loss, torch.Tensor)


def test_spectral_convergence_loss_forward(sample_audio, target_audio):
    """Test SpectralConvergenceLoss forward pass."""
    spectral_convergence_loss = SpectralConvergenceLoss()
    loss = spectral_convergence_loss(sample_audio, target_audio)
    assert isinstance(loss, torch.Tensor)


def test_log_stft_magnitude_loss_forward(sample_audio, target_audio):
    """Test LogSTFTMagnitudeLoss forward pass."""
    log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
    loss = log_stft_magnitude_loss(sample_audio, target_audio)
    assert isinstance(loss, torch.Tensor)


def test_stft_loss_forward(sample_audio, target_audio):
    """Test STFTLoss forward pass."""
    stft_loss = STFTLoss()
    loss = stft_loss(sample_audio, target_audio)
    assert isinstance(loss, torch.Tensor)


def test_multi_resolution_stft_loss_forward(sample_audio, target_audio):
    """Test MultiResolutionSTFTLoss forward pass."""
    multi_resolution_stft_loss = MultiResolutionSTFTLoss()
    loss = multi_resolution_stft_loss(sample_audio, target_audio)
    assert isinstance(loss, torch.Tensor)


def test_mel_spectrogram_loss_forward(sample_audio, target_audio):
    """Test MelSpectrogramLoss forward pass."""
    mel_spectrogram_loss = MelSpectrogramLoss()
    loss = mel_spectrogram_loss(sample_audio, target_audio)
    assert isinstance(loss, torch.Tensor)


def test_feature_matching_loss_forward(sample_audio, target_audio):
    """Test FeatureMatchingLoss forward pass."""

    # Assuming a dummy model for feature extraction
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return x

    model = DummyModel()
    feature_matching_loss = FeatureMatchingLoss(model, layers=[0])
    loss = feature_matching_loss(sample_audio, target_audio)
    assert isinstance(loss, torch.Tensor)


def test_audio_contrastive_loss_forward(sample_audio, target_audio):
    """Test AudioContrastiveLoss forward pass."""
    audio_contrastive_loss = AudioContrastiveLoss()
    features = torch.randn(10, 128)  # 10 samples with 128-dimensional features
    labels = torch.randint(0, 2, (10,))  # Binary labels for contrastive loss
    loss = audio_contrastive_loss(features, labels)
    assert isinstance(loss, torch.Tensor)
