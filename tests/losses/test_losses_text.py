"""Tests for the text losses module with comprehensive coverage."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from kaira.losses.text import (
    CrossEntropyLoss,
    LabelSmoothingLoss,
    CosineSimilarityLoss,
    Word2VecLoss,
)


@pytest.fixture
def sample_logits():
    """Fixture for creating sample logits tensor."""
    return torch.tensor([
        [0.1, 0.9, 0.2, 0.3, 0.4],  # Highest prob on class 1
        [0.8, 0.2, 0.3, 0.4, 0.1],  # Highest prob on class 0
        [0.1, 0.2, 0.1, 0.1, 0.9]   # Highest prob on class 4
    ], requires_grad=True)  # Add requires_grad to test gradient flow


@pytest.fixture
def sample_targets():
    """Fixture for creating sample target tensor."""
    return torch.tensor([1, 0, 4])  # Target classes for the samples


@pytest.fixture
def sample_embeddings():
    """Fixture for creating sample embedding tensors."""
    return torch.randn(5, 64, requires_grad=True)  # 5 samples with 64-dim embeddings


@pytest.fixture
def sample_target_embeddings():
    """Fixture for creating sample target embedding tensors."""
    return torch.randn(5, 64, requires_grad=True)  # 5 target samples with 64-dim embeddings


class TestCrossEntropyLoss:
    """Test suite for CrossEntropyLoss."""
    
    def test_forward_basic(self, sample_logits, sample_targets):
        """Test basic forward pass with default parameters."""
        loss_fn = CrossEntropyLoss()
        loss = loss_fn(sample_logits, sample_targets)
        
        # Check that the loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.requires_grad
        
        # Compare with PyTorch's built-in CrossEntropyLoss
        torch_ce = nn.CrossEntropyLoss()(sample_logits, sample_targets)
        assert torch.isclose(loss, torch_ce)
    
    def test_with_class_weights(self, sample_logits, sample_targets):
        """Test with custom class weights."""
        weights = torch.tensor([0.2, 0.8, 0.3, 0.5, 1.0])
        loss_fn = CrossEntropyLoss(weight=weights)
        loss = loss_fn(sample_logits, sample_targets)
        
        # Compare with PyTorch's built-in weighted CrossEntropyLoss
        torch_ce = nn.CrossEntropyLoss(weight=weights)(sample_logits, sample_targets)
        assert torch.isclose(loss, torch_ce)
    
    def test_with_ignore_index(self, sample_logits):
        """Test with ignore_index parameter."""
        # Create targets with an ignore index
        targets = torch.tensor([1, -100, 4])
        
        loss_fn = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(sample_logits, targets)
        
        # Compare with PyTorch's built-in CrossEntropyLoss with ignore_index
        torch_ce = nn.CrossEntropyLoss(ignore_index=-100)(sample_logits, targets)
        assert torch.isclose(loss, torch_ce)
    
    def test_with_label_smoothing(self, sample_logits, sample_targets):
        """Test with label_smoothing parameter."""
        smoothing = 0.1
        loss_fn = CrossEntropyLoss(label_smoothing=smoothing)
        loss = loss_fn(sample_logits, sample_targets)
        
        # Compare with PyTorch's built-in CrossEntropyLoss with label_smoothing
        torch_ce = nn.CrossEntropyLoss(label_smoothing=smoothing)(sample_logits, sample_targets)
        assert torch.isclose(loss, torch_ce)
    
    def test_gradient_flow(self, sample_logits, sample_targets):
        """Test that gradients flow properly."""
        loss_fn = CrossEntropyLoss()
        loss = loss_fn(sample_logits, sample_targets)
        
        # Check that we can backpropagate through the loss
        loss.backward()
        
        # Check that gradients were computed
        assert sample_logits.grad is not None


class TestLabelSmoothingLoss:
    """Test suite for LabelSmoothingLoss."""
    
    def test_forward_basic(self, sample_logits, sample_targets):
        """Test basic forward pass with default parameters."""
        loss_fn = LabelSmoothingLoss(smoothing=0.1, classes=5)
        loss = loss_fn(sample_logits, sample_targets)
        
        # Check that the loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
    
    def test_different_smoothing_values(self, sample_logits, sample_targets):
        """Test with different smoothing values."""
        # No smoothing
        loss_fn_0 = LabelSmoothingLoss(smoothing=0.0, classes=5)
        loss_0 = loss_fn_0(sample_logits, sample_targets)
        
        # Some smoothing
        loss_fn_01 = LabelSmoothingLoss(smoothing=0.1, classes=5)
        loss_01 = loss_fn_01(sample_logits, sample_targets)
        
        # More smoothing
        loss_fn_02 = LabelSmoothingLoss(smoothing=0.2, classes=5)
        loss_02 = loss_fn_02(sample_logits, sample_targets)
        
        # With no smoothing, the loss should be similar to cross entropy
        ce_loss = nn.CrossEntropyLoss()(sample_logits, sample_targets)
        assert torch.isclose(loss_0, ce_loss, rtol=1e-4)
        
        # More smoothing should give different loss values
        assert loss_0.item() != loss_01.item()
        assert loss_01.item() != loss_02.item()
    
    def test_gradient_flow(self, sample_logits, sample_targets):
        """Test that gradients flow properly."""
        loss_fn = LabelSmoothingLoss(smoothing=0.1, classes=5)
        loss = loss_fn(sample_logits, sample_targets)
        
        # Check that we can backpropagate through the loss
        loss.backward()
        
        # Check that gradients were computed
        assert sample_logits.grad is not None


class TestCosineSimilarityLoss:
    """Test suite for CosineSimilarityLoss."""
    
    def test_forward_basic(self, sample_embeddings, sample_target_embeddings):
        """Test basic forward pass with default parameters."""
        loss_fn = CosineSimilarityLoss()
        loss = loss_fn(sample_embeddings, sample_target_embeddings)
        
        # Check that the loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
    
    def test_with_different_margins(self, sample_embeddings, sample_target_embeddings):
        """Test with different margin values."""
        # No margin
        loss_fn_0 = CosineSimilarityLoss(margin=0.0)
        loss_0 = loss_fn_0(sample_embeddings, sample_target_embeddings)
        
        # Small margin
        loss_fn_05 = CosineSimilarityLoss(margin=0.5)
        loss_05 = loss_fn_05(sample_embeddings, sample_target_embeddings)
        
        # Large margin
        loss_fn_1 = CosineSimilarityLoss(margin=1.0)
        loss_1 = loss_fn_1(sample_embeddings, sample_target_embeddings)
        
        # Larger margin should produce larger or equal loss
        # (unless all similarities are already above the margin)
        assert loss_0.item() <= loss_05.item()
        assert loss_05.item() <= loss_1.item()
    
    def test_identical_embeddings(self):
        """Test with identical embeddings (should give zero loss with margin=0)."""
        embeddings = torch.randn(5, 64)
        
        # With margin=0, identical embeddings should give zero loss
        loss_fn = CosineSimilarityLoss(margin=0.0)
        loss = loss_fn(embeddings, embeddings)
        
        assert torch.isclose(loss, torch.tensor(0.0))
    
    def test_orthogonal_embeddings(self):
        """Test with orthogonal embeddings."""
        # Create a pair of orthogonal embeddings
        emb1 = torch.tensor([[0.0, 1.0]], requires_grad=True)  # [1, 2]
        emb2 = torch.tensor([[1.0, 0.0]], requires_grad=True)  # [1, 2]
        
        # Cosine similarity should be 0 for orthogonal vectors
        cs = F.cosine_similarity(emb1, emb2)
        assert torch.isclose(cs, torch.tensor(0.0))
        
        # With margin=0.5, loss should be 0.5
        loss_fn = CosineSimilarityLoss(margin=0.5)
        loss = loss_fn(emb1, emb2)
        assert torch.isclose(loss, torch.tensor(0.5))
        
        # With margin=0.0, loss should be 0.0
        loss_fn = CosineSimilarityLoss(margin=0.0)
        loss = loss_fn(emb1, emb2)
        assert torch.isclose(loss, torch.tensor(0.0))


class TestWord2VecLoss:
    """Test suite for Word2VecLoss."""
    
    def test_forward_basic(self):
        """Test basic forward pass."""
        batch_size = 5
        vocab_size = 100
        embedding_dim = 64
        
        input_idx = torch.randint(0, vocab_size, (batch_size,))
        output_idx = torch.randint(0, vocab_size, (batch_size,))
        
        loss_fn = Word2VecLoss(embedding_dim=embedding_dim, vocab_size=vocab_size)
        loss = loss_fn(input_idx, output_idx)
        
        # Check that the loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
    
    def test_with_different_negative_samples(self):
        """Test with different numbers of negative samples."""
        batch_size = 5
        vocab_size = 100
        embedding_dim = 64
        
        input_idx = torch.randint(0, vocab_size, (batch_size,))
        output_idx = torch.randint(0, vocab_size, (batch_size,))
        
        # Test with different numbers of negative samples
        loss_fn_1 = Word2VecLoss(embedding_dim=embedding_dim, vocab_size=vocab_size, n_negatives=1)
        loss_1 = loss_fn_1(input_idx, output_idx)
        
        loss_fn_10 = Word2VecLoss(embedding_dim=embedding_dim, vocab_size=vocab_size, n_negatives=10)
        loss_10 = loss_fn_10(input_idx, output_idx)
        
        # Losses should be different due to different number of negative samples
        assert loss_1.item() != loss_10.item()
    
    def test_embedding_shapes(self):
        """Test the shapes of the embeddings."""
        vocab_size = 100
        embedding_dim = 64
        
        loss_fn = Word2VecLoss(embedding_dim=embedding_dim, vocab_size=vocab_size)
        
        # Check embedding shapes
        assert loss_fn.in_embed.weight.shape == (vocab_size, embedding_dim)
        assert loss_fn.out_embed.weight.shape == (vocab_size, embedding_dim)
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through both embeddings."""
        batch_size = 5
        vocab_size = 100
        embedding_dim = 64
        
        input_idx = torch.randint(0, vocab_size, (batch_size,))
        output_idx = torch.randint(0, vocab_size, (batch_size,))
        
        loss_fn = Word2VecLoss(embedding_dim=embedding_dim, vocab_size=vocab_size)
        
        # Set requires_grad to track gradients
        loss_fn.in_embed.weight.requires_grad = True
        loss_fn.out_embed.weight.requires_grad = True
        
        # Forward pass
        loss = loss_fn(input_idx, output_idx)
        
        # Backward pass
        loss.backward()
        
        # Check that both embedding matrices received gradients
        assert loss_fn.in_embed.weight.grad is not None
        assert loss_fn.out_embed.weight.grad is not None
        
        # Gradients should be sparse (only for used indices)
        # Create mask of used indices for input embeddings
        in_used = torch.zeros(vocab_size, dtype=torch.bool)
        in_used[input_idx] = True
        
        # For indices that weren't used, gradients should be zero
        assert not loss_fn.in_embed.weight.grad[~in_used].abs().sum() > 0