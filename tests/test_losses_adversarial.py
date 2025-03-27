import pytest
import torch

# filepath: kaira/losses/test_adversarial.py

from ..losses.adversarial import (
    FeatureMatchingLoss,
    HingeLoss,
    LSGANLoss,
    R1GradientPenalty,
    VanillaGANLoss,
    WassersteinGANLoss,
)


@pytest.fixture
def predictions():
    return torch.tensor([0.6, 0.7, 0.8], requires_grad=True)


@pytest.fixture
def real_features():
    # Create a list of features of different dimensions
    return [
        torch.randn(3, 16, 16, 16, requires_grad=True),
        torch.randn(3, 8, 8, 8, requires_grad=True),
        torch.randn(3, 4, 4, 4, requires_grad=True)
    ]


@pytest.fixture
def fake_features():
    # Create a list of features of different dimensions
    return [
        torch.randn(3, 16, 16, 16, requires_grad=True),
        torch.randn(3, 8, 8, 8, requires_grad=True),
        torch.randn(3, 4, 4, 4, requires_grad=True)
    ]


@pytest.fixture
def real_data():
    return torch.randn(3, 3, 32, 32, requires_grad=True)


@pytest.fixture
def none_gradient_data():
    # Data that will produce None gradient in R1 penalty
    return torch.ones(3, 3, 2, 2, requires_grad=True)


@pytest.fixture
def real_outputs():
    return torch.tensor([0.9, 0.8, 0.7], requires_grad=True)


def test_lsgan_loss_forward_discriminator_real(predictions):
    loss_fn = LSGANLoss()
    loss = loss_fn.forward(predictions, is_real=True, for_discriminator=True)
    assert isinstance(loss, torch.Tensor)
    # LSGAN loss for real samples in discriminator should be (D(x) - 1)^2
    expected_loss = torch.mean((predictions - 1) ** 2)
    assert torch.allclose(loss, expected_loss)


def test_lsgan_loss_forward_discriminator_fake(predictions):
    loss_fn = LSGANLoss()
    loss = loss_fn.forward(predictions, is_real=False, for_discriminator=True)
    assert isinstance(loss, torch.Tensor)
    # LSGAN loss for fake samples in discriminator should be D(G(z))^2
    expected_loss = torch.mean(predictions ** 2)
    assert torch.allclose(loss, expected_loss)


def test_lsgan_loss_forward_generator(predictions):
    loss_fn = LSGANLoss()
    loss = loss_fn.forward(predictions, is_real=False, for_discriminator=False)
    assert isinstance(loss, torch.Tensor)
    # LSGAN loss for generator should be (D(G(z)) - 1)^2
    expected_loss = torch.mean((predictions - 1) ** 2)
    assert torch.allclose(loss, expected_loss)


def test_wasserstein_gan_loss_forward_discriminator_real(predictions):
    loss_fn = WassersteinGANLoss()
    loss = loss_fn.forward(predictions, is_real=True, for_discriminator=True)
    assert isinstance(loss, torch.Tensor)
    # WGAN loss for real samples in discriminator should be -D(x)
    expected_loss = -torch.mean(predictions)
    assert torch.allclose(loss, expected_loss)


def test_wasserstein_gan_loss_forward_discriminator_fake(predictions):
    loss_fn = WassersteinGANLoss()
    loss = loss_fn.forward(predictions, is_real=False, for_discriminator=True)
    assert isinstance(loss, torch.Tensor)
    # WGAN loss for fake samples in discriminator should be D(G(z))
    expected_loss = torch.mean(predictions)
    assert torch.allclose(loss, expected_loss)


def test_wasserstein_gan_loss_forward_generator(predictions):
    loss_fn = WassersteinGANLoss()
    loss = loss_fn.forward(predictions, is_real=False, for_discriminator=False)
    assert isinstance(loss, torch.Tensor)
    # WGAN loss for generator should be -D(G(z))
    expected_loss = -torch.mean(predictions)
    assert torch.allclose(loss, expected_loss)


def test_hinge_loss_forward_discriminator_real(predictions):
    loss_fn = HingeLoss()
    loss = loss_fn.forward(predictions, is_real=True, for_discriminator=True)
    assert isinstance(loss, torch.Tensor)
    # Hinge loss for real samples in discriminator should be max(0, 1-D(x))
    expected_loss = torch.relu(1.0 - predictions).mean()
    assert torch.allclose(loss, expected_loss)


def test_hinge_loss_forward_discriminator_fake(predictions):
    loss_fn = HingeLoss()
    loss = loss_fn.forward(predictions, is_real=False, for_discriminator=True)
    assert isinstance(loss, torch.Tensor)
    # Hinge loss for fake samples in discriminator should be max(0, 1+D(G(z)))
    expected_loss = torch.relu(1.0 + predictions).mean()
    assert torch.allclose(loss, expected_loss)


def test_hinge_loss_forward_generator(predictions):
    loss_fn = HingeLoss()
    loss = loss_fn.forward(predictions, is_real=False, for_discriminator=False)
    assert isinstance(loss, torch.Tensor)
    # Hinge loss for generator should be -D(G(z))
    expected_loss = -predictions.mean()
    assert torch.allclose(loss, expected_loss)


def test_feature_matching_loss_with_different_feature_sizes(real_features, fake_features):
    loss_fn = FeatureMatchingLoss()
    loss = loss_fn(real_features, fake_features)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Should be a scalar


def test_feature_matching_loss_backward(real_features, fake_features):
    """Test that feature matching loss is differentiable."""
    loss_fn = FeatureMatchingLoss()
    loss = loss_fn(real_features, fake_features)
    
    # Verify we can backpropagate through the loss
    loss.backward()
    
    # Check that gradients were computed for fake features (which should be updated)
    for feature in fake_features:
        assert feature.grad is not None


def test_r1_gradient_penalty_backward(real_data, real_outputs):
    """Test that R1 gradient penalty is differentiable."""
    loss_fn = R1GradientPenalty(gamma=10.0)
    penalty = loss_fn(real_data, real_outputs)
    
    # Verify we can backpropagate through the penalty
    penalty.backward()
    
    # Check that gradients were computed
    assert real_data.grad is not None


def test_r1_gradient_penalty_with_none_gradient(none_gradient_data):
    """Test R1 gradient penalty when gradient might be None."""
    # Create a scalar output that won't depend on all input dimensions
    constant_output = torch.tensor(1.0, requires_grad=True)
    
    loss_fn = R1GradientPenalty(gamma=10.0)
    penalty = loss_fn(none_gradient_data, constant_output)
    
    # Should return zero tensor on device when gradient is None
    assert penalty.item() == 0.0


def test_r1_gradient_penalty_with_different_gamma():
    """Test R1 gradient penalty with different gamma values."""
    # Create test data
    real_data = torch.randn(2, 2, 2, 2, requires_grad=True)
    real_outputs = torch.sum(real_data * real_data)
    
    # Test with different gamma values
    gamma1 = 1.0
    gamma2 = 20.0
    
    loss_fn1 = R1GradientPenalty(gamma=gamma1)
    loss_fn2 = R1GradientPenalty(gamma=gamma2)
    
    penalty1 = loss_fn1(real_data, real_outputs)
    penalty2 = loss_fn2(real_data, real_outputs)
    
    # The ratio of penalties should be the same as the ratio of gammas
    ratio = penalty2 / penalty1
    assert abs(ratio.item() - (gamma2 / gamma1)) < 1e-5