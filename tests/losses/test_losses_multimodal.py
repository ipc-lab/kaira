"""Unified comprehensive tests for multimodal loss functions."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from kaira.losses.base import BaseLoss
from kaira.losses.multimodal import (
    AlignmentLoss,
    CMCLoss,
    ContrastiveLoss,
    InfoNCELoss,
    TripletLoss,
)


@pytest.fixture
def embedding_pairs():
    """Fixture providing pairs of embeddings for multimodal testing."""
    # Create two sets of normalized embedding vectors
    batch_size = 8
    embed_dim = 64

    # Create paired embeddings
    torch.manual_seed(42)  # For reproducibility
    embeddings1 = torch.randn(batch_size, embed_dim)
    embeddings2 = torch.randn(batch_size, embed_dim)

    return embeddings1, embeddings2


@pytest.fixture
def triplet_data():
    """Fixture providing triplet data for triplet loss testing."""
    batch_size = 8
    embed_dim = 64

    torch.manual_seed(42)  # For reproducibility

    # Create anchor, positive and negative embeddings
    anchors = torch.randn(batch_size, embed_dim)
    # Positives are similar to anchors but with some noise
    positives = anchors + 0.1 * torch.randn(batch_size, embed_dim)
    # Negatives are more different from anchors
    negatives = -anchors + 0.5 * torch.randn(batch_size, embed_dim)

    anchors.requires_grad_(True)  # Enable gradient computation
    positives.requires_grad_(True)  # Enable gradient computation
    negatives.requires_grad_(True)  # Enable gradient computation

    # Create labels (same label for anchor and positive, different for negative)
    labels = torch.arange(batch_size)

    return anchors, positives, negatives, labels


class SimpleProjection(BaseLoss):
    """Simple projection network for CMCLoss testing."""

    def __init__(self, input_dim=64, output_dim=32):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim))

    def forward(self, x):
        return self.projection(x)


class TestContrastiveLoss:
    """Tests for ContrastiveLoss."""

    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Default initialization
        loss_fn = ContrastiveLoss()
        assert loss_fn.margin == 0.2
        assert loss_fn.temperature == 0.07

        # Custom initialization
        loss_fn = ContrastiveLoss(margin=0.5, temperature=0.1)
        assert loss_fn.margin == 0.5
        assert loss_fn.temperature == 0.1

    def test_forward_paired_data(self, embedding_pairs):
        embeddings1, embeddings2 = embedding_pairs
        embeddings1.requires_grad_(True)  # Enable gradient computation
        embeddings2.requires_grad_(True)  # Enable gradient computation
        loss_fn = ContrastiveLoss()
        loss = loss_fn(embeddings1, embeddings2)

        # Check loss is a scalar tensor with grad_fn
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.grad_fn is not None  # Has gradient function

        # Loss should be positive
        assert loss.item() > 0

    def test_forward_with_labels(self, embedding_pairs):
        embeddings1, embeddings2 = embedding_pairs
        embeddings1.requires_grad_(True)  # Enable gradient computation
        embeddings2.requires_grad_(True)  # Enable gradient computation
        loss_fn = ContrastiveLoss()
        labels = torch.tensor([0, 1, 0, 3, 4, 5, 6, 7])
        loss = loss_fn(embeddings1, embeddings2, labels)

        # Check loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.grad_fn is not None
        assert loss.item() > 0

    def test_gradient_flow(self, embedding_pairs):
        """Test gradient flow through the contrastive loss."""
        embeddings1, embeddings2 = embedding_pairs

        # Make embeddings require gradients
        embeddings1.requires_grad_(True)
        embeddings2.requires_grad_(True)

        loss_fn = ContrastiveLoss()
        loss = loss_fn(embeddings1, embeddings2)

        # Backpropagate
        loss.backward()

        # Check gradients exist
        assert embeddings1.grad is not None
        assert embeddings2.grad is not None

        # Check gradients are not zero
        assert not torch.allclose(embeddings1.grad, torch.zeros_like(embeddings1.grad))
        assert not torch.allclose(embeddings2.grad, torch.zeros_like(embeddings2.grad))

    def test_similar_dissimilar_pairs(self, embedding_pairs):
        """Test ContrastiveLoss with similar and dissimilar pairs."""
        anchor, positive = embedding_pairs

        # Create dissimilar pairs by shuffling the positive samples
        idx = torch.randperm(anchor.size(0))
        negative = positive[idx]

        # Initialize loss
        loss_fn = ContrastiveLoss(margin=0.5)

        # Similar pairs should have low loss
        similar_loss = loss_fn(anchor, positive)

        # Create labels that indicate all pairs are dissimilar
        dissimilar_labels = torch.zeros(anchor.size(0), device=anchor.device)

        # Dissimilar pairs should have higher loss
        dissimilar_loss = loss_fn(anchor, negative, dissimilar_labels)

        assert similar_loss.item() > 0
        assert dissimilar_loss.item() > 0


class TestTripletLoss:
    """Tests for TripletLoss."""

    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Default initialization
        loss_fn = TripletLoss()
        assert loss_fn.margin == 0.3
        assert loss_fn.distance == "cosine"

        # Custom initialization
        loss_fn = TripletLoss(margin=0.5, distance="euclidean")
        assert loss_fn.margin == 0.5
        assert loss_fn.distance == "euclidean"

    def test_forward_with_explicit_negatives_cosine(self, triplet_data):
        anchors, positives, negatives, _ = triplet_data
        loss_fn = TripletLoss(distance="cosine")
        loss = loss_fn(anchors, positives, negatives)

        # Check loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.grad_fn is not None
        assert loss.item() >= 0  # Triplet loss is always non-negative

    def test_forward_with_explicit_negatives_euclidean(self, triplet_data):
        anchors, positives, negatives, _ = triplet_data
        anchors.requires_grad_(True)  # Enable gradient computation
        positives.requires_grad_(True)  # Enable gradient computation
        negatives.requires_grad_(True)  # Enable gradient computation
        loss_fn = TripletLoss(distance="euclidean")
        loss = loss_fn(anchors, positives, negatives)

        # Check loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.grad_fn is not None
        assert loss.item() >= 0

    def test_forward_with_online_mining_cosine(self, triplet_data):
        """Test forward pass with online mining using cosine distance."""
        anchors, positives, _, labels = triplet_data
        # Explicitly enable gradient computation for inputs
        anchors = anchors.clone().detach().requires_grad_(True)
        positives = positives.clone().detach().requires_grad_(True)

        loss_fn = TripletLoss(distance="cosine")

        # Forward pass with online mining
        loss = loss_fn(anchors, positives, labels=labels)

        # Check loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.grad_fn is not None
        assert loss.item() >= 0

    def test_forward_with_online_mining_euclidean(self, triplet_data):
        """Test forward pass with online mining using euclidean distance."""
        anchors, positives, _, labels = triplet_data
        # Explicitly enable gradient computation for inputs
        anchors = anchors.clone().detach().requires_grad_(True)
        positives = positives.clone().detach().requires_grad_(True)

        loss_fn = TripletLoss(distance="euclidean")

        # Forward pass with online mining
        loss = loss_fn(anchors, positives, labels=labels)

        # Check loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.grad_fn is not None
        assert loss.item() >= 0

    def test_error_when_no_negatives_or_labels(self, triplet_data):
        """Test that error is raised when neither negatives nor labels are provided."""
        anchors, positives, _, _ = triplet_data
        loss_fn = TripletLoss()

        # Should raise ValueError with specific message when neither negatives nor labels are provided
        with pytest.raises(ValueError, match="Either negative samples or labels must be provided"):
            loss_fn(anchors, positives)

    def test_error_for_both_distance_metrics(self, triplet_data):
        """Test that error is raised for both cosine and euclidean metrics when neither negatives
        nor labels are provided."""
        anchors, positives, _, _ = triplet_data

        # Test with cosine distance
        loss_fn_cosine = TripletLoss(distance="cosine")
        with pytest.raises(ValueError, match="Either negative samples or labels must be provided"):
            loss_fn_cosine(anchors, positives)

        # Test with euclidean distance
        loss_fn_euclidean = TripletLoss(distance="euclidean")
        with pytest.raises(ValueError, match="Either negative samples or labels must be provided"):
            loss_fn_euclidean(anchors, positives)

    def test_no_valid_negatives_case(self, triplet_data):
        """Test case when no valid negatives can be found (all same label)."""
        anchors, positives, _, _ = triplet_data
        loss_fn = TripletLoss()

        # All samples have the same label
        same_labels = torch.zeros(anchors.size(0), dtype=torch.long)

        # Should return mean of positive distances
        loss = loss_fn(anchors, positives, labels=same_labels)

        # Check that loss calculation doesn't crash
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_no_valid_negatives_euclidean(self, triplet_data):
        """Test case when no valid negatives can be found with euclidean distance."""
        anchors, positives, _, _ = triplet_data
        loss_fn = TripletLoss(distance="euclidean")

        same_labels = torch.zeros(anchors.size(0), dtype=torch.long)

        loss = loss_fn(anchors, positives, labels=same_labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_invalid_distance_metric(self):
        """Test TripletLoss with invalid distance metric."""
        with pytest.raises(ValueError):
            TripletLoss(distance="invalid_distance")


class TestInfoNCELoss:
    """Tests for InfoNCELoss."""

    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Default initialization
        loss_fn = InfoNCELoss()
        assert loss_fn.temperature == 0.07

        # Custom initialization
        loss_fn = InfoNCELoss(temperature=0.1)
        assert loss_fn.temperature == 0.1

    def test_forward_without_queue(self, embedding_pairs):
        """Test forward pass without external negative queue."""
        query, key = embedding_pairs
        query.requires_grad_(True)  # Enable gradient computation
        key.requires_grad_(True)  # Enable gradient computation
        loss_fn = InfoNCELoss()

        # Forward pass using batch samples as negatives
        loss = loss_fn(query, key)

        # Check loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.grad_fn is not None
        assert loss.item() > 0

    def test_forward_with_queue(self, embedding_pairs):
        """Test forward pass with external negative queue."""
        query, key = embedding_pairs
        query.requires_grad_(True)  # Enable gradient computation
        key.requires_grad_(True)  # Enable gradient computation
        loss_fn = InfoNCELoss()

        # Create a negative queue
        queue_size = 32
        embed_dim = query.shape[1]
        queue = torch.randn(queue_size, embed_dim, requires_grad=True)  # Enable gradient computation

        # Forward pass with external negative queue
        loss = loss_fn(query, key, queue)

        # Check loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.grad_fn is not None
        assert loss.item() > 0

    def test_gradient_flow(self, embedding_pairs):
        """Test gradient flow through the InfoNCE loss."""
        query, key = embedding_pairs

        # Make embeddings require gradients
        query.requires_grad_(True)
        key.requires_grad_(True)

        loss_fn = InfoNCELoss()
        loss = loss_fn(query, key)

        # Backpropagate
        loss.backward()

        # Check gradients exist
        assert query.grad is not None
        assert key.grad is not None

        # Check gradients are not zero
        assert not torch.allclose(query.grad, torch.zeros_like(query.grad))
        assert not torch.allclose(key.grad, torch.zeros_like(key.grad))

    def test_temperature_scaling(self, embedding_pairs):
        """Test that different temperature values affect the loss."""
        query, key = embedding_pairs

        # Compare losses with different temperature values
        loss_fn_low_temp = InfoNCELoss(temperature=0.01)
        loss_fn_high_temp = InfoNCELoss(temperature=1.0)

        loss_low_temp = loss_fn_low_temp(query, key)
        loss_high_temp = loss_fn_high_temp(query, key)

        # Different temperatures should give different loss values
        assert loss_low_temp.item() != loss_high_temp.item()

    def test_with_mask(self, embedding_pairs):
        """Test InfoNCELoss with a masking matrix for valid pairs."""
        emb1, emb2 = embedding_pairs
        batch_size = emb1.size(0)

        # Create a mask where only diagonal elements are valid pairs
        mask = torch.eye(batch_size)

        # Create loss function
        loss_fn = InfoNCELoss()

        loss = loss_fn(emb1, emb2, mask=mask)

        # If supported, verify basic properties
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_no_negatives_case(self, query_features, key_features):
        """Test the case where there are no negative pairs for InfoNCELoss."""
        query_features.requires_grad_(True)
        key_features.requires_grad_(True)

        batch_size = query_features.size(0)

        # Create a mask where ALL pairs are positive
        # This will trigger the branch where no negatives are found
        mask = torch.ones(batch_size, batch_size)

        loss_fn = InfoNCELoss()
        loss = loss_fn(query_features, key_features, mask=mask)

        # Manually compute what we expect: -l_pos.mean()
        # Normalize features first as the implementation does
        query_norm = F.normalize(query_features, p=2, dim=1)
        key_norm = F.normalize(key_features, p=2, dim=1)

        # Compute similarities
        similarities = torch.einsum("nc,kc->nk", [query_norm, key_norm])

        # We expect l_pos to be the max similarity for each query
        # Since all pairs are positive, this would be the max value in each row
        l_pos = similarities.max(dim=1, keepdim=True)[0]
        expected_loss = -l_pos.mean()

        # Verify the loss matches what we expect
        assert torch.isclose(loss, expected_loss)

        # Check gradient flow
        loss.backward()
        assert query_features.grad is not None
        assert key_features.grad is not None


class TestCMCLoss:
    """Tests for Cross-Modal Consistency Loss."""

    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Default initialization
        loss_fn = CMCLoss()
        assert loss_fn.lambda_cmc == 1.0

        # Custom initialization
        loss_fn = CMCLoss(lambda_cmc=0.5)
        assert loss_fn.lambda_cmc == 0.5

    def test_forward(self, embedding_pairs):
        """Test forward pass with projection heads."""
        x1, x2 = embedding_pairs
        input_dim = x1.shape[1]
        output_dim = 32

        # Create projection heads
        proj1 = SimpleProjection(input_dim, output_dim)
        proj2 = SimpleProjection(input_dim, output_dim)

        loss_fn = CMCLoss()

        # Forward pass
        loss = loss_fn(x1, x2, proj1, proj2)

        # Check loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.grad_fn is not None
        assert loss.item() > 0

    def test_gradient_flow(self, embedding_pairs):
        """Test gradient flow through the CMC loss and projections."""
        x1, x2 = embedding_pairs
        input_dim = x1.shape[1]
        output_dim = 32

        # Make embeddings require gradients
        x1.requires_grad_(True)
        x2.requires_grad_(True)

        # Create projection heads
        proj1 = SimpleProjection(input_dim, output_dim)
        proj2 = SimpleProjection(input_dim, output_dim)

        loss_fn = CMCLoss()
        loss = loss_fn(x1, x2, proj1, proj2)

        # Backpropagate
        loss.backward()

        # Check gradients exist
        assert x1.grad is not None
        assert x2.grad is not None

        # Check gradients are not zero
        assert not torch.allclose(x1.grad, torch.zeros_like(x1.grad))
        assert not torch.allclose(x2.grad, torch.zeros_like(x2.grad))

    def test_with_different_weight(self, embedding_pairs):
        """Test that lambda_cmc parameter properly scales the loss."""
        x1, x2 = embedding_pairs
        input_dim = x1.shape[1]
        output_dim = 32

        # Create projection heads
        proj1 = SimpleProjection(input_dim, output_dim)
        proj2 = SimpleProjection(input_dim, output_dim)

        # Compare losses with different lambda values
        loss_fn1 = CMCLoss(lambda_cmc=1.0)
        loss_fn2 = CMCLoss(lambda_cmc=2.0)

        loss1 = loss_fn1(x1, x2, proj1, proj2)
        loss2 = loss_fn2(x1, x2, proj1, proj2)

        # Loss with lambda=2.0 should be approximately twice the loss with lambda=1.0
        assert abs(loss2.item() - 2.0 * loss1.item()) < 1e-5

    def test_with_identity_projection(self, embedding_pairs):
        """Test CMCLoss with identity projection."""
        x1, x2 = embedding_pairs

        # Create an identity projection
        class IdentityProjection(torch.nn.Module):
            def forward(self, x):
                return x

        proj1 = IdentityProjection()
        proj2 = IdentityProjection()

        # Initialize loss
        loss_fn = CMCLoss(lambda_cmc=1.0)

        # Compute loss
        loss = loss_fn(x1, x2, proj1, proj2)

        assert loss.item() > 0  # Loss should be positive


class TestAlignmentLoss:
    """Tests for Alignment Loss."""

    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Default initialization
        loss_fn = AlignmentLoss()
        assert loss_fn.alignment_type == "l2"

        # Custom initialization
        loss_fn_l1 = AlignmentLoss(alignment_type="l1")
        assert loss_fn_l1.alignment_type == "l1"

        loss_fn_cosine = AlignmentLoss(alignment_type="cosine")
        assert loss_fn_cosine.alignment_type == "cosine"

    def test_l2_alignment(self, embedding_pairs):
        """Test L2 alignment loss."""
        x1, x2 = embedding_pairs
        x1.requires_grad_(True)  # Enable gradient computation
        x2.requires_grad_(True)  # Enable gradient computation
        loss_fn = AlignmentLoss(alignment_type="l2")

        # Forward pass
        loss = loss_fn(x1, x2)

        # Check loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.grad_fn is not None
        assert loss.item() > 0

        # Verify it's L2 loss by comparing with F.mse_loss
        expected_loss = torch.nn.functional.mse_loss(x1, x2)
        assert torch.isclose(loss, expected_loss)

    def test_l1_alignment(self, embedding_pairs):
        """Test L1 alignment loss."""
        x1, x2 = embedding_pairs
        x1.requires_grad_(True)  # Enable gradient computation
        x2.requires_grad_(True)  # Enable gradient computation
        loss_fn = AlignmentLoss(alignment_type="l1")

        # Forward pass
        loss = loss_fn(x1, x2)

        # Check loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.grad_fn is not None
        assert loss.item() > 0

        # Verify it's L1 loss by comparing with F.l1_loss
        expected_loss = torch.nn.functional.l1_loss(x1, x2)
        assert torch.isclose(loss, expected_loss)

    def test_cosine_alignment(self, embedding_pairs):
        """Test cosine alignment loss."""
        x1, x2 = embedding_pairs
        x1.requires_grad_(True)  # Enable gradient computation
        x2.requires_grad_(True)  # Enable gradient computation
        loss_fn = AlignmentLoss(alignment_type="cosine")

        # Forward pass
        loss = loss_fn(x1, x2)

        # Check loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.grad_fn is not None
        assert 0.0 <= loss.item() <= 2.0  # Cosine distance range

    def test_invalid_alignment_type(self):
        """Test that error is raised for invalid alignment type."""
        with pytest.raises(ValueError):
            AlignmentLoss(alignment_type="invalid")

    def test_perfect_alignment(self):
        """Test loss is zero for perfectly aligned embeddings."""
        # Create identical embeddings
        x = torch.randn(8, 64)

        # Test with all alignment types
        for alignment_type in ["l1", "l2", "cosine"]:
            loss_fn = AlignmentLoss(alignment_type=alignment_type)
            loss = loss_fn(x, x)
            assert loss.item() < 1e-6  # Should be very close to zero

    def test_different_projections(self, embedding_pairs):
        """Test AlignmentLoss with different projection dimensions."""
        x1, x2 = embedding_pairs

        loss_no_proj = AlignmentLoss(projection_dim=None)  # No projection
        loss_small_proj = AlignmentLoss(projection_dim=32)  # Smaller projection

        # Compute losses
        value_no_proj = loss_no_proj(x1, x2)
        value_small_proj = loss_small_proj(x1, x2)

        # Verify results
        assert isinstance(value_no_proj, torch.Tensor)
        assert isinstance(value_small_proj, torch.Tensor)

    def test_unsupported_alignment_type_init(self):
        """Test AlignmentLoss with unsupported alignment type during initialization."""
        with pytest.raises(ValueError, match=r"Unsupported alignment type: .*"):
            AlignmentLoss(alignment_type="invalid_type")

    def test_unsupported_alignment_type_forward(self):
        """Test AlignmentLoss with unsupported alignment type during forward pass."""
        # Create a loss instance with a valid type initially
        loss_fn = AlignmentLoss(alignment_type="l2")
        batch_size = 8
        embed_dim = 32

        x1 = torch.randn(batch_size, embed_dim)
        x2 = torch.randn(batch_size, embed_dim)

        # Manually set an invalid alignment type to trigger the error in forward
        loss_fn.alignment_type = "invalid_type"

        with pytest.raises(ValueError, match=r"Unsupported alignment type: .*"):
            loss_fn(x1, x2)

    def test_unsupported_alignment_type_forward_case(self):
        """Test specifically the error case when alignment_type is invalid during forward pass."""
        # Create a loss instance with a valid type initially
        loss_fn = AlignmentLoss(alignment_type="l2")
        batch_size = 8
        embed_dim = 32

        x1 = torch.randn(batch_size, embed_dim)
        x2 = torch.randn(batch_size, embed_dim)

        # Manually set an invalid alignment type to trigger the error in forward
        loss_fn.alignment_type = "invalid_type"

        # This should trigger the specific error case we want to cover
        with pytest.raises(ValueError, match=r"Unsupported alignment type: invalid_type"):
            loss_fn(x1, x2)


# Common fixtures
@pytest.fixture
def random_tensor():
    def _random_tensor(shape, normalize=False):
        tensor = torch.randn(*shape)
        if normalize:
            tensor = torch.nn.functional.normalize(tensor, p=2, dim=1)
        return tensor

    return _random_tensor


# Fixtures specifically for InfoNCELoss tests
@pytest.fixture
def query_features(random_tensor):
    batch_size = 8
    dim = 64
    return random_tensor((batch_size, dim))


@pytest.fixture
def key_features(random_tensor):
    batch_size = 8
    dim = 64
    return random_tensor((batch_size, dim))


# Test classes for each loss
