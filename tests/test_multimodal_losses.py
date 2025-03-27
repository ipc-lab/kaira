import pytest
import torch

from kaira.losses.multimodal import (
    ContrastiveLoss,
    TripletLoss,
    InfoNCELoss,
    CMCLoss,
    AlignmentLoss
)


@pytest.fixture
def embedding_data():
    """Fixture to create sample embedding data."""
    batch_size = 8
    embedding_dim = 16
    
    # Create embeddings
    anchor = torch.randn(batch_size, embedding_dim)
    positive = anchor + 0.1 * torch.randn(batch_size, embedding_dim)  # Similar to anchor
    negative = torch.randn(batch_size, embedding_dim)  # Different from anchor
    
    # Create labels
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    
    return anchor, positive, negative, labels


def test_contrastive_loss_similar_pairs(embedding_data):
    """Test ContrastiveLoss with similar pairs."""
    anchor, positive, _, _ = embedding_data
    
    # Initialize loss
    loss_fn = ContrastiveLoss(margin=0.5)
    
    # Similar pairs should have low loss
    loss = loss_fn(anchor, positive)
    
    assert loss.item() > 0
    assert loss.item() < 0.5  # Loss should be small for similar pairs


def test_contrastive_loss_dissimilar_pairs(embedding_data):
    """Test ContrastiveLoss with dissimilar pairs."""
    anchor, _, negative, _ = embedding_data
    
    # Initialize loss
    loss_fn = ContrastiveLoss(margin=0.5)
    
    # Create labels that indicate all pairs are dissimilar
    dissimilar_labels = torch.zeros(anchor.size(0), device=anchor.device)
    
    # Dissimilar pairs should have higher loss
    loss = loss_fn(anchor, negative, dissimilar_labels)
    
    assert loss.item() > 0


def test_triplet_loss_cosine(embedding_data):
    """Test TripletLoss with cosine similarity."""
    anchor, positive, negative, _ = embedding_data
    
    # Initialize loss with cosine similarity
    loss_fn = TripletLoss(margin=0.2, distance="cosine")
    
    # Compute loss
    loss = loss_fn(anchor, positive, negative)
    
    assert loss.item() >= 0  # Loss should be non-negative


def test_triplet_loss_euclidean(embedding_data):
    """Test TripletLoss with euclidean distance."""
    anchor, positive, negative, _ = embedding_data
    
    # Initialize loss with euclidean distance
    loss_fn = TripletLoss(margin=0.2, distance="euclidean")
    
    # Compute loss
    loss = loss_fn(anchor, positive, negative)
    
    assert loss.item() >= 0  # Loss should be non-negative


def test_triplet_loss_online_mining(embedding_data):
    """Test TripletLoss with online mining using labels."""
    anchor, positive, _, labels = embedding_data
    
    # Initialize loss
    loss_fn = TripletLoss(margin=0.2)
    
    # Compute loss with online mining
    loss = loss_fn(anchor, positive, labels=labels)
    
    assert loss.item() >= 0  # Loss should be non-negative


def test_triplet_loss_invalid_distance():
    """Test TripletLoss with invalid distance metric."""
    with pytest.raises(ValueError):
        TripletLoss(distance="invalid_distance")


def test_infonce_loss_basic(embedding_data):
    """Test basic InfoNCELoss functionality."""
    query, key, _, _ = embedding_data
    
    # Initialize loss
    loss_fn = InfoNCELoss(temperature=0.1)
    
    # Compute loss
    loss = loss_fn(query, key)
    
    assert loss.item() > 0  # Loss should be positive


def test_infonce_loss_with_queue(embedding_data):
    """Test InfoNCELoss with memory queue."""
    query, key, _, _ = embedding_data
    
    # Create a memory queue
    queue_size = 16
    embedding_dim = query.shape[1]
    queue = torch.randn(queue_size, embedding_dim)
    
    # Initialize loss
    loss_fn = InfoNCELoss(temperature=0.1)
    
    # Compute loss with queue
    loss = loss_fn(query, key, queue)
    
    assert loss.item() > 0  # Loss should be positive


def test_cmc_loss(embedding_data):
    """Test CMCLoss functionality."""
    x1, x2, _, _ = embedding_data
    
    # Create projection networks (mock with identity)
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


def test_alignment_loss(embedding_data):
    """Test AlignmentLoss functionality."""
    x1, x2, _, _ = embedding_data
    
    # Initialize loss
    loss_fn = AlignmentLoss()
    
    # Compute loss
    loss = loss_fn(x1, x2)
    
    assert loss.item() >= 0  # Loss should be non-negative
