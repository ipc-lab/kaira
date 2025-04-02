import pytest
import torch

from kaira.losses.audio import AudioContrastiveLoss


@pytest.fixture
def audio_data():
    """Fixture for creating sample audio batch."""
    # Create a batch of 4 audio samples, each 1 second at 16kHz
    return torch.randn(4, 16000)


@pytest.fixture
def audio_embedding():
    """Fixture for creating sample audio embeddings."""
    # Create 4 audio embeddings of dimension 128
    return torch.randn(4, 128)


def test_audio_contrastive_loss_default_params():
    """Test AudioContrastiveLoss with default parameters."""
    loss_fn = AudioContrastiveLoss()
    assert loss_fn.temperature == 0.1
    assert loss_fn.normalize is True
    assert loss_fn.reduction == "mean"


def test_audio_contrastive_loss_custom_params():
    """Test AudioContrastiveLoss with custom parameters."""
    loss_fn = AudioContrastiveLoss(temperature=0.5, normalize=False, reduction="sum")
    assert loss_fn.temperature == 0.5
    assert loss_fn.normalize is False
    assert loss_fn.reduction == "sum"


def test_audio_contrastive_loss_precomputed_embeddings(audio_embedding):
    """Test AudioContrastiveLoss with precomputed embeddings."""
    loss_fn = AudioContrastiveLoss()

    # Same embeddings for anchor and positive (perfect similarity)
    loss = loss_fn(audio_embedding, audio_embedding)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar output
    assert loss.item() > 0  # Loss should be positive


def test_audio_contrastive_loss_with_view_maker(audio_data):
    """Test AudioContrastiveLoss with view maker."""
    loss_fn = AudioContrastiveLoss()

    # Create a simple view maker (just add noise)
    def view_maker(x):
        return x + torch.randn_like(x) * 0.01

    # Test with view maker
    loss = loss_fn(audio_data, audio_data, view_maker=view_maker)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar output
    assert loss.item() > 0  # Loss should be positive


def test_audio_contrastive_loss_with_projector(audio_embedding):
    """Test AudioContrastiveLoss with projector."""
    # Create a simple projector network
    projector = torch.nn.Linear(128, 64)

    loss_fn = AudioContrastiveLoss()

    # Test with projector
    loss = loss_fn(audio_embedding, audio_embedding, projector=projector)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar output
    assert loss.item() > 0  # Loss should be positive


def test_audio_contrastive_loss_with_labels(audio_embedding):
    """Test AudioContrastiveLoss with labels for defining positive pairs."""
    loss_fn = AudioContrastiveLoss()

    # Create sample labels: samples with same label are positive pairs
    labels = torch.tensor([0, 0, 1, 1])

    # Test with labels
    loss = loss_fn(audio_embedding, audio_embedding, labels=labels)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar output
    assert loss.item() > 0  # Loss should be positive
