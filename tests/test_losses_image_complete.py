import pytest
import torch
from kaira.losses.image import ElasticLoss, StyleLoss


@pytest.fixture
def sample_preds():
    """Fixture for creating sample predictions tensor."""
    return torch.randn(2, 3, 64, 64)


@pytest.fixture
def sample_targets():
    """Fixture for creating sample targets tensor."""
    return torch.randn(2, 3, 64, 64)


def test_elastic_loss_different_alpha_beta():
    """Test ElasticLoss with different alpha and beta values."""
    x = torch.randn(2, 3, 64, 64)
    y = torch.randn(2, 3, 64, 64)
    
    # Test with different alpha and beta values
    loss1 = ElasticLoss(alpha=0.5, beta=0.5)
    loss2 = ElasticLoss(alpha=0.8, beta=0.2)
    loss3 = ElasticLoss(alpha=0.2, beta=0.8)
    
    value1 = loss1(x, y)
    value2 = loss2(x, y)
    value3 = loss3(x, y)
    
    assert isinstance(value1, torch.Tensor)
    assert isinstance(value2, torch.Tensor)
    assert isinstance(value3, torch.Tensor)
    
    # Different parameter settings should give different loss values
    assert value1.item() != value2.item()
    assert value2.item() != value3.item()
    assert value1.item() != value3.item()


def test_elastic_loss_edge_cases():
    """Test ElasticLoss edge cases."""
    x = torch.randn(2, 3, 64, 64)
    y = torch.randn(2, 3, 64, 64)
    
    # With alpha=1, beta=0, it should act like L1 loss
    loss_l1 = ElasticLoss(alpha=1.0, beta=0.0)
    # With alpha=0, beta=1, it should act like a quadratic loss
    loss_l2 = ElasticLoss(alpha=0.0, beta=1.0)
    
    l1_value = loss_l1(x, y)
    l2_value = loss_l2(x, y)
    
    # Compare with direct computation
    direct_l1 = torch.mean(torch.abs(x - y))
    direct_l2 = torch.mean((x - y) ** 2)
    
    assert torch.isclose(l1_value, direct_l1)
    assert torch.isclose(l2_value, direct_l2)


def test_style_loss_with_gram_matrices(sample_preds, sample_targets):
    """Test StyleLoss using precomputed Gram matrices."""
    # Create mock gram matrices
    gram_pred = torch.randn(2, 8, 8)  # [batch, C, C]
    gram_target = torch.randn(2, 8, 8)
    
    loss_fn = StyleLoss(apply_gram=False)  # Don't compute gram matrices internally
    loss = loss_fn(gram_pred, gram_target)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar output
    assert loss >= 0  # Loss should be non-negative


def test_style_loss_normalization(sample_preds, sample_targets):
    """Test StyleLoss with and without normalization."""
    # Test with normalization on
    loss_normalized = StyleLoss(normalize=True)
    value_norm = loss_normalized(sample_preds, sample_targets)
    
    # Test with normalization off
    loss_unnormalized = StyleLoss(normalize=False)
    value_unnorm = loss_unnormalized(sample_preds, sample_targets)
    
    # The values should be different with different normalization settings
    assert value_norm.item() != value_unnorm.item()


def test_style_loss_custom_layer_weights():
    """Test StyleLoss with custom layer weights."""
    # Create mock feature maps from VGG
    feature_preds = {
        'relu1_2': torch.randn(2, 64, 32, 32),
        'relu2_2': torch.randn(2, 128, 16, 16),
        'relu3_3': torch.randn(2, 256, 8, 8)
    }
    
    feature_targets = {
        'relu1_2': torch.randn(2, 64, 32, 32),
        'relu2_2': torch.randn(2, 128, 16, 16),
        'relu3_3': torch.randn(2, 256, 8, 8)
    }
    
    # Define custom layer weights
    layer_weights = {
        'relu1_2': 0.1,
        'relu2_2': 0.2,
        'relu3_3': 0.7
    }
    
    loss_fn = StyleLoss(layer_weights=layer_weights, apply_gram=False)
    
    # Mock the _extract_features method to return our predefined features
    loss_fn._extract_features = lambda x: feature_preds if torch.equal(x, torch.tensor(1.0)) else feature_targets
    
    # Call forward with dummy tensors
    loss = loss_fn(torch.tensor(1.0), torch.tensor(2.0))
    
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar output
    assert loss >= 0  # Loss should be non-negative
