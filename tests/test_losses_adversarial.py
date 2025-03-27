import pytest
import torch
from kaira.losses.adversarial import VanillaGANLoss, LSGANLoss, WassersteinGANLoss, HingeLoss, FeatureMatchingLoss, R1GradientPenalty

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
def real_outputs():
    return torch.tensor([0.9, 0.8, 0.7], requires_grad=True)

def test_vanilla_gan_loss_discriminator(real_logits, fake_logits):
    loss_fn = VanillaGANLoss()
    loss = loss_fn.forward_discriminator(real_logits, fake_logits)
    assert isinstance(loss, torch.Tensor)

def test_vanilla_gan_loss_generator(fake_logits):
    loss_fn = VanillaGANLoss()
    loss = loss_fn.forward_generator(fake_logits)
    assert isinstance(loss, torch.Tensor)

def test_lsgan_loss_discriminator(real_logits, fake_logits):
    loss_fn = LSGANLoss()
    loss = loss_fn.forward_discriminator(real_logits, fake_logits)
    assert isinstance(loss, torch.Tensor)

def test_lsgan_loss_generator(fake_logits):
    loss_fn = LSGANLoss()
    loss = loss_fn.forward_generator(fake_logits)
    assert isinstance(loss, torch.Tensor)

def test_wasserstein_gan_loss_discriminator(real_logits, fake_logits):
    loss_fn = WassersteinGANLoss()
    loss = loss_fn.forward_discriminator(real_logits, fake_logits)
    assert isinstance(loss, torch.Tensor)

def test_wasserstein_gan_loss_generator(fake_logits):
    loss_fn = WassersteinGANLoss()
    loss = loss_fn.forward_generator(fake_logits)
    assert isinstance(loss, torch.Tensor)

def test_hinge_loss_discriminator(real_logits, fake_logits):
    loss_fn = HingeLoss()
    loss = loss_fn.forward_discriminator(real_logits, fake_logits)
    assert isinstance(loss, torch.Tensor)

def test_hinge_loss_generator(fake_logits):
    loss_fn = HingeLoss()
    loss = loss_fn.forward_generator(fake_logits)
    assert isinstance(loss, torch.Tensor)

def test_feature_matching_loss(real_data, real_outputs):
    loss_fn = FeatureMatchingLoss()
    loss = loss_fn(real_data, real_outputs)
    assert isinstance(loss, torch.Tensor)

def test_r1_gradient_penalty(real_data, real_outputs):
    loss_fn = R1GradientPenalty()
    loss = loss_fn(real_data, real_outputs)
    assert isinstance(loss, torch.Tensor)
