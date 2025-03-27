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
    
    # Create a model with proper layers for feature extraction
    class LayeredModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([
                torch.nn.Conv1d(1, 4, kernel_size=3, padding=1),
                torch.nn.Conv1d(4, 8, kernel_size=3, padding=1)
            ])
            
        def forward(self, x):
            # Make sure input has proper shape for Conv1d (batch, channels, length)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            
            features = []
            for layer in self.layers:
                x = layer(x)
                features.append(x)
            return x, features
    
    model = LayeredModel()
    feature_matching_loss = FeatureMatchingLoss(model, layers=[0, 1])
    
    # Pass through model first to get features
    _, real_features = model(sample_audio)
    _, fake_features = model(target_audio)
    
    # Then pass the features to the loss
    loss = feature_matching_loss(real_features, fake_features)
    
    assert isinstance(loss, torch.Tensor)
    
    # Test the forward method directly with audio tensors
    direct_loss = feature_matching_loss(sample_audio, target_audio)
    assert isinstance(direct_loss, torch.Tensor)
    
    # Test with custom weights
    weighted_loss = FeatureMatchingLoss(model, layers=[0, 1], weights=[0.3, 0.7])
    weighted_result = weighted_loss(sample_audio, target_audio)
    assert isinstance(weighted_result, torch.Tensor)


def test_audio_contrastive_loss_forward(sample_audio, target_audio):
    """Test AudioContrastiveLoss forward pass."""
    audio_contrastive_loss = AudioContrastiveLoss()
    features = torch.randn(10, 128)  # 10 samples with 128-dimensional features
    labels = torch.randint(0, 2, (10,))  # Binary labels for contrastive loss
    loss = audio_contrastive_loss(features, labels=labels)
    assert isinstance(loss, torch.Tensor)


def test_audio_contrastive_loss_with_projector():
    """Test AudioContrastiveLoss with a projector network."""
    audio_contrastive_loss = AudioContrastiveLoss()
    features = torch.randn(10, 256)  # 10 samples with 256-dimensional features
    
    # Create a simple projector network
    class Projector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(256, 128)
            
        def forward(self, x):
            return self.fc(x)
    
    projector = Projector()
    loss = audio_contrastive_loss(features, projector=projector)
    assert isinstance(loss, torch.Tensor)


def test_audio_contrastive_loss_with_view_maker():
    """Test AudioContrastiveLoss with a view maker function."""
    audio_contrastive_loss = AudioContrastiveLoss()
    features = torch.randn(10, 128)  # 10 samples with 128-dimensional features
    
    # Create a simple view maker function
    def view_maker(x):
        return x + 0.1 * torch.randn_like(x)
    
    loss = audio_contrastive_loss(features, view_maker=view_maker)
    assert isinstance(loss, torch.Tensor)
    
    # Test with both view_maker and target
    target = torch.randn(10, 128)
    loss_with_target = audio_contrastive_loss(features, target=target, view_maker=view_maker)
    assert isinstance(loss_with_target, torch.Tensor)


def test_audio_contrastive_loss_reduction_methods():
    """Test different reduction methods in AudioContrastiveLoss."""
    features = torch.randn(10, 128)
    labels = torch.randint(0, 2, (10,))
    
    # Test mean reduction (default)
    mean_loss = AudioContrastiveLoss(reduction="mean")
    mean_result = mean_loss(features, labels=labels)
    assert isinstance(mean_result, torch.Tensor)
    assert mean_result.dim() == 0  # scalar output
    
    # Test sum reduction
    sum_loss = AudioContrastiveLoss(reduction="sum")
    sum_result = sum_loss(features, labels=labels)
    assert isinstance(sum_result, torch.Tensor)
    assert sum_result.dim() == 0  # scalar output
    
    # Test none reduction
    none_loss = AudioContrastiveLoss(reduction="none")
    none_result = none_loss(features, labels=labels)
    assert isinstance(none_result, torch.Tensor)
    assert none_result.shape == (10,)  # One loss value per sample
