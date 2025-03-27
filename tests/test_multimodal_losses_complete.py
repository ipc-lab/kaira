import pytest
import torch

from kaira.losses.multimodal import AlignmentLoss, InfoNCELoss


@pytest.fixture
def embedding_data():
    """Fixture to create sample embedding data."""
    batch_size = 8
    emb_dim = 64

    embeddings1 = torch.randn(batch_size, emb_dim)
    embeddings2 = torch.randn(batch_size, emb_dim)

    # Normalize embeddings
    embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
    embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1)

    return embeddings1, embeddings2


def test_infonce_loss_temperature(embedding_data):
    """Test InfoNCELoss with different temperature values."""
    emb1, emb2 = embedding_data

    # Test with different temperature values
    loss_high_temp = InfoNCELoss(temperature=1.0)
    loss_low_temp = InfoNCELoss(temperature=0.01)

    value_high = loss_high_temp(emb1, emb2)
    value_low = loss_low_temp(emb1, emb2)

    # Lower temperature typically results in higher loss values
    # as it makes the distribution more peaked
    assert isinstance(value_high, torch.Tensor)
    assert isinstance(value_low, torch.Tensor)
    assert value_high.item() != value_low.item()


def test_infonce_loss_negative_sampling(embedding_data):
    """Test InfoNCELoss with different negative sampling techniques."""
    emb1, emb2 = embedding_data

    # Test with explicit negatives
    extra_negatives = torch.randn(4, emb1.size(1))  # 4 extra negative samples
    extra_negatives = torch.nn.functional.normalize(extra_negatives, p=2, dim=1)

    loss_fn = InfoNCELoss()
    loss_with_extra = loss_fn(emb1, emb2, extra_negatives)

    # Compare with default (in-batch negatives only)
    loss_default = loss_fn(emb1, emb2)

    assert isinstance(loss_with_extra, torch.Tensor)
    assert isinstance(loss_default, torch.Tensor)
    # With more negatives, the task is harder, so loss should be different
    assert loss_with_extra.item() != loss_default.item()


def test_infonce_loss_with_mask(embedding_data):
    """Test InfoNCELoss with a masking matrix for valid pairs."""
    emb1, emb2 = embedding_data
    batch_size = emb1.size(0)

    # Create a mask where only diagonal elements are valid pairs
    mask = torch.eye(batch_size)

    loss_fn = InfoNCELoss()
    loss = loss_fn(emb1, emb2, mask=mask)

    # Compare with default (diagonal mask is default behavior)
    loss_default = loss_fn(emb1, emb2)

    assert isinstance(loss, torch.Tensor)
    # Should be close since we're using the same mask pattern
    assert torch.isclose(loss, loss_default, rtol=1e-4)


def test_alignment_loss_different_projections(embedding_data):
    """Test AlignmentLoss with different projection dimensions."""
    emb1, emb2 = embedding_data
    emb1.size(1)

    # Test with different projection dimensions
    loss_no_proj = AlignmentLoss(projection_dim=None)  # No projection
    loss_small_proj = AlignmentLoss(projection_dim=32)  # Smaller projection
    loss_large_proj = AlignmentLoss(projection_dim=128)  # Larger projection

    value_no_proj = loss_no_proj(emb1, emb2)
    value_small_proj = loss_small_proj(emb1, emb2)
    value_large_proj = loss_large_proj(emb1, emb2)

    assert isinstance(value_no_proj, torch.Tensor)
    assert isinstance(value_small_proj, torch.Tensor)
    assert isinstance(value_large_proj, torch.Tensor)
    # Different projections should give different results
    assert value_no_proj.item() != value_small_proj.item() or value_small_proj.item() != value_large_proj.item()


def test_alignment_loss_with_weights(embedding_data):
    """Test AlignmentLoss with custom weights for alignment and uniformity."""
    emb1, emb2 = embedding_data

    # Test with different weight configurations
    loss_align_heavy = AlignmentLoss(alignment_weight=1.0, uniformity_weight=0.0)
    loss_uniform_heavy = AlignmentLoss(alignment_weight=0.0, uniformity_weight=1.0)
    loss_balanced = AlignmentLoss(alignment_weight=0.5, uniformity_weight=0.5)

    value_align = loss_align_heavy(emb1, emb2)
    value_uniform = loss_uniform_heavy(emb1, emb2)
    value_balanced = loss_balanced(emb1, emb2)

    assert isinstance(value_align, torch.Tensor)
    assert isinstance(value_uniform, torch.Tensor)
    assert isinstance(value_balanced, torch.Tensor)
    # Different weight configurations should give different results
    assert value_align.item() != value_uniform.item()
    # Balanced should be somewhere in between
    min_val = min(value_align.item(), value_uniform.item())
    max_val = max(value_align.item(), value_uniform.item())
    assert min_val <= value_balanced.item() <= max_val or pytest.approx(min_val) == value_balanced.item() or pytest.approx(max_val) == value_balanced.item()
