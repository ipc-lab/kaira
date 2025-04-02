import pytest
import torch

from kaira.losses.adversarial import (
    FeatureMatchingLoss,
    HingeLoss,
    LSGANLoss,
    R1GradientPenalty,
    VanillaGANLoss,
    WassersteinGANLoss,
)


@pytest.fixture
def real_logits():
    return torch.tensor([0.9, 0.8, 0.7])


@pytest.fixture
def fake_logits():
    return torch.tensor([0.1, 0.2, 0.3])


@pytest.fixture
def real_data():
    return torch.randn(3, 3, 32, 32, requires_grad=True)


@pytest.fixture
def fake_data():
    return torch.randn(3, 3, 32, 32, requires_grad=True)


@pytest.fixture
def real_outputs():
    return torch.tensor([0.9, 0.8, 0.7], requires_grad=True)


@pytest.fixture
def fake_outputs():
    return torch.tensor([0.1, 0.2, 0.3], requires_grad=True)


def test_vanilla_gan_loss_backward_compatibility(real_logits, fake_logits):
    """Test backward compatibility of VanillaGANLoss."""
    loss_fn = VanillaGANLoss()
    
    # Use the new API with separate methods instead of mode parameter
    d_loss = loss_fn.discriminator_loss(real_logits, fake_logits)
    g_loss = loss_fn.generator_loss(fake_logits)
    
    assert isinstance(d_loss, torch.Tensor)
    assert isinstance(g_loss, torch.Tensor)


def test_lsgan_loss_backward_compatibility(real_logits, fake_logits):
    """Test backward compatibility of LSGANLoss."""
    loss_fn = LSGANLoss()
    
    # Use the new API with separate methods instead of mode parameter
    d_loss = loss_fn.discriminator_loss(real_logits, fake_logits)
    g_loss = loss_fn.generator_loss(fake_logits)
    
    assert isinstance(d_loss, torch.Tensor)
    assert isinstance(g_loss, torch.Tensor)


def test_wasserstein_gan_loss_backward_compatibility(real_logits, fake_logits):
    """Test backward compatibility of WassersteinGANLoss."""
    loss_fn = WassersteinGANLoss()
    
    # Use the new API with separate methods instead of mode parameter
    d_loss = loss_fn.discriminator_loss(real_logits, fake_logits)
    g_loss = loss_fn.generator_loss(fake_logits)
    
    assert isinstance(d_loss, torch.Tensor)
    assert isinstance(g_loss, torch.Tensor)


def test_hinge_loss_backward_compatibility(real_logits, fake_logits):
    """Test backward compatibility of HingeLoss."""
    loss_fn = HingeLoss()
    
    # Use the new API with separate methods instead of mode parameter
    d_loss = loss_fn.discriminator_loss(real_logits, fake_logits)
    g_loss = loss_fn.generator_loss(fake_logits)
    
    assert isinstance(d_loss, torch.Tensor)
    assert isinstance(g_loss, torch.Tensor)


def test_feature_matching_loss_with_different_features(real_data, fake_data):
    """Test FeatureMatchingLoss with multiple feature levels."""

    # Create a simple discriminator that returns multiple features
    class MultiFeatureDiscriminator(torch.nn.Module):
        def forward(self, x):
            features = [x, x * 0.5, x * 0.25]
            return features[-1], features

    discriminator = MultiFeatureDiscriminator()
    loss_fn = FeatureMatchingLoss()

    # Get features from the discriminator
    _, real_features = discriminator(real_data)
    _, fake_features = discriminator(fake_data)

    # Calculate loss
    loss = loss_fn(real_features, fake_features)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar output


def test_r1_gradient_penalty_with_discriminator(real_data):
    """Test R1GradientPenalty with a discriminator network."""
    # Create a simple discriminator that returns a scalar output
    class SimpleDiscriminator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = torch.nn.Flatten()
            # Adjust in_features to match the flattened size of the input tensor
            self.linear = torch.nn.Linear(3 * 32 * 32, 1)

        def forward(self, x):
            x = self.flatten(x)
            return self.linear(x)

    discriminator = SimpleDiscriminator()
    
    loss_fn = R1GradientPenalty(gamma=10.0)

    # Call discriminator directly to get real_outputs
    real_outputs = discriminator(real_data)
    
    # Calculate penalty using outputs that we already have
    penalty = loss_fn(real_data, real_outputs)
    
    assert isinstance(penalty, torch.Tensor)
    assert penalty >= 0  # Penalty should be non-negative
