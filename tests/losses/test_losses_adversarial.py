"""Tests for the adversarial losses module with comprehensive coverage."""

import torch
import torch.nn.functional as F
import pytest

from kaira.losses.adversarial import (
    VanillaGANLoss, 
    LSGANLoss,
    WassersteinGANLoss,
    HingeLoss,
    FeatureMatchingLoss,
    R1GradientPenalty
)


class TestVanillaGANLoss:
    """Test suite for VanillaGANLoss."""

    def test_forward_method(self):
        """Test the forward method directly."""
        loss_fn = VanillaGANLoss()
        pred = torch.randn(5, 1)
        
        # Test for real labels
        real_loss = loss_fn(pred, is_real=True)
        assert isinstance(real_loss, torch.Tensor)
        
        # Test for fake labels
        fake_loss = loss_fn(pred, is_real=False)
        assert isinstance(fake_loss, torch.Tensor)
    
    def test_reduction_methods(self):
        """Test different reduction methods."""
        # Test mean reduction
        loss_fn_mean = VanillaGANLoss(reduction='mean')
        # Test sum reduction
        loss_fn_sum = VanillaGANLoss(reduction='sum')
        # Test none reduction
        loss_fn_none = VanillaGANLoss(reduction='none')
        
        pred = torch.randn(5, 1)
        
        # Verify each reduction produces expected output shape
        mean_loss = loss_fn_mean(pred, is_real=True)
        assert mean_loss.shape == torch.Size([])
        
        sum_loss = loss_fn_sum(pred, is_real=True)
        assert sum_loss.shape == torch.Size([])
        
        none_loss = loss_fn_none(pred, is_real=True)
        assert none_loss.shape == pred.shape

    def test_forward_discriminator(self):
        """Test the forward_discriminator method specifically."""
        loss_fn = VanillaGANLoss()
        real_logits = torch.randn(5, 1)
        fake_logits = torch.randn(5, 1)
        
        # Test discriminator loss
        d_loss = loss_fn.forward_discriminator(real_logits, fake_logits)
        assert isinstance(d_loss, torch.Tensor)
        assert d_loss.shape == torch.Size([])
        
        # Verify loss calculation
        expected_real_loss = F.binary_cross_entropy_with_logits(
            real_logits, torch.ones_like(real_logits))
        expected_fake_loss = F.binary_cross_entropy_with_logits(
            fake_logits, torch.zeros_like(fake_logits))
        expected_total_loss = expected_real_loss + expected_fake_loss
        
        assert torch.isclose(d_loss, expected_total_loss)
    
    def test_forward_generator(self):
        """Test the forward_generator method specifically."""
        loss_fn = VanillaGANLoss()
        fake_logits = torch.randn(5, 1)
        
        # Test generator loss
        g_loss = loss_fn.forward_generator(fake_logits)
        assert isinstance(g_loss, torch.Tensor)
        assert g_loss.shape == torch.Size([])
        
        # Verify loss calculation
        expected_loss = F.binary_cross_entropy_with_logits(
            fake_logits, torch.ones_like(fake_logits))
        
        assert torch.isclose(g_loss, expected_loss)


class TestLSGANLoss:
    """Test suite for LSGANLoss."""

    def test_forward_method(self):
        """Test the forward method directly."""
        loss_fn = LSGANLoss()
        pred = torch.randn(5, 1)
        
        # Test for real data (discriminator)
        loss_d_real = loss_fn(pred, is_real=True, for_discriminator=True)
        assert isinstance(loss_d_real, torch.Tensor)
        
        # Test for fake data (discriminator)
        loss_d_fake = loss_fn(pred, is_real=False, for_discriminator=True)
        assert isinstance(loss_d_fake, torch.Tensor)
        
        # Test for generator
        loss_g = loss_fn(pred, is_real=False, for_discriminator=False)
        assert isinstance(loss_g, torch.Tensor)
    
    def test_reduction_methods(self):
        """Test different reduction methods."""
        # Test mean reduction (default)
        loss_fn = LSGANLoss(reduction='mean')
        
        pred = torch.randn(5, 1)
        loss = loss_fn(pred, is_real=True)
        assert loss.dim() == 0  # Scalar tensor

    def test_forward_discriminator(self):
        """Test the forward_discriminator method specifically."""
        loss_fn = LSGANLoss()
        real_pred = torch.randn(5, 1)
        fake_pred = torch.randn(5, 1)
        
        # Test discriminator loss
        d_loss = loss_fn.forward_discriminator(real_pred, fake_pred)
        assert isinstance(d_loss, torch.Tensor)
        assert d_loss.shape == torch.Size([])
        
        # Verify loss calculation
        expected_real_loss = torch.mean((real_pred - 1) ** 2)
        expected_fake_loss = torch.mean(fake_pred**2)
        expected_total_loss = (expected_real_loss + expected_fake_loss) * 0.5
        
        assert torch.isclose(d_loss, expected_total_loss)
    
    def test_forward_generator(self):
        """Test the forward_generator method specifically."""
        loss_fn = LSGANLoss()
        fake_pred = torch.randn(5, 1)
        
        # Test generator loss
        g_loss = loss_fn.forward_generator(fake_pred)
        assert isinstance(g_loss, torch.Tensor)
        assert g_loss.shape == torch.Size([])
        
        # Verify loss calculation
        expected_loss = torch.mean((fake_pred - 1) ** 2)
        
        assert torch.isclose(g_loss, expected_loss)


class TestWassersteinGANLoss:
    """Test suite for WassersteinGANLoss."""

    def test_forward_method(self):
        """Test the forward method directly."""
        loss_fn = WassersteinGANLoss()
        pred = torch.randn(5, 1)
        
        # Test for real data (discriminator)
        loss_d_real = loss_fn(pred, is_real=True, for_discriminator=True)
        assert isinstance(loss_d_real, torch.Tensor)
        
        # Test for fake data (discriminator)
        loss_d_fake = loss_fn(pred, is_real=False, for_discriminator=True)
        assert isinstance(loss_d_fake, torch.Tensor)
        
        # Test for generator
        loss_g = loss_fn(pred, is_real=False, for_discriminator=False)
        assert isinstance(loss_g, torch.Tensor)
        
    def test_loss_values(self):
        """Test expected loss values for specific inputs."""
        loss_fn = WassersteinGANLoss()
        
        # All ones for real
        ones = torch.ones(5, 1)
        # All zeros for fake
        zeros = torch.zeros(5, 1)
        
        # Discriminator should minimize: -(E[D(real)] - E[D(fake)])
        d_loss = loss_fn.forward_discriminator(ones, zeros)
        assert d_loss == -1.0  # -mean(1) + mean(0) = -1
        
        # Generator should minimize: -E[D(fake)]
        g_loss = loss_fn.forward_generator(zeros)
        assert g_loss == 0.0  # -mean(0) = 0


class TestHingeLoss:
    """Test suite for HingeLoss."""

    def test_forward_method(self):
        """Test the forward method directly."""
        loss_fn = HingeLoss()
        pred = torch.randn(5, 1)
        
        # Test for real data (discriminator)
        loss_d_real = loss_fn(pred, is_real=True, for_discriminator=True)
        assert isinstance(loss_d_real, torch.Tensor)
        
        # Test for fake data (discriminator)
        loss_d_fake = loss_fn(pred, is_real=False, for_discriminator=True)
        assert isinstance(loss_d_fake, torch.Tensor)
        
        # Test for generator
        loss_g = loss_fn(pred, is_real=False, for_discriminator=False)
        assert isinstance(loss_g, torch.Tensor)
        
    def test_loss_values(self):
        """Test expected loss values for specific inputs."""
        loss_fn = HingeLoss()
        
        # Values greater than 1 (should give 0 real loss)
        high_vals = torch.ones(5, 1) * 2.0
        
        # Values less than -1 (should give 0 fake loss)
        low_vals = torch.ones(5, 1) * -2.0
        
        # Real loss should be relu(1-pred).mean()
        real_loss = loss_fn(high_vals, is_real=True)
        assert real_loss == 0.0
        
        # Fake loss for discriminator should be relu(1+pred).mean()
        fake_d_loss = loss_fn(low_vals, is_real=False)
        assert fake_d_loss == 0.0

    def test_forward_discriminator(self):
        """Test the forward_discriminator method specifically."""
        loss_fn = HingeLoss()
        real_pred = torch.randn(5, 1)
        fake_pred = torch.randn(5, 1)
        
        # Test discriminator loss
        d_loss = loss_fn.forward_discriminator(real_pred, fake_pred)
        assert isinstance(d_loss, torch.Tensor)
        assert d_loss.shape == torch.Size([])
        
        # Verify loss calculation
        expected_real_loss = F.relu(1.0 - real_pred).mean()
        expected_fake_loss = F.relu(1.0 + fake_pred).mean()
        expected_total_loss = expected_real_loss + expected_fake_loss
        
        assert torch.isclose(d_loss, expected_total_loss)
    
    def test_forward_generator(self):
        """Test the forward_generator method specifically."""
        loss_fn = HingeLoss()
        fake_pred = torch.randn(5, 1)
        
        # Test generator loss
        g_loss = loss_fn.forward_generator(fake_pred)
        assert isinstance(g_loss, torch.Tensor)
        assert g_loss.shape == torch.Size([])
        
        # Verify loss calculation
        expected_loss = -fake_pred.mean()
        
        assert torch.isclose(g_loss, expected_loss)


class TestFeatureMatchingLoss:
    """Test suite for FeatureMatchingLoss."""

    def test_forward_with_single_feature(self):
        """Test feature matching with single feature."""
        loss_fn = FeatureMatchingLoss()
        real_features = [torch.randn(4, 10)]  # batch_size=4, feature_dim=10
        fake_features = [torch.randn(4, 10)]
        
        loss = loss_fn(real_features, fake_features)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar tensor
    
    def test_forward_with_multiple_features(self):
        """Test feature matching with multiple feature layers."""
        loss_fn = FeatureMatchingLoss()
        real_features = [torch.randn(4, 8), torch.randn(4, 16), torch.randn(4, 32)]
        fake_features = [torch.randn(4, 8), torch.randn(4, 16), torch.randn(4, 32)]
        
        loss = loss_fn(real_features, fake_features)
        assert isinstance(loss, torch.Tensor)
    
    def test_identical_features(self):
        """Test with identical real and fake features (should give zero loss)."""
        loss_fn = FeatureMatchingLoss()
        features = [torch.randn(4, 8), torch.randn(4, 16)]
        
        loss = loss_fn(features, features)
        assert loss.item() == 0.0


class TestR1GradientPenalty:
    """Test suite for R1GradientPenalty."""

    def test_forward_with_different_gamma(self):
        """Test R1 gradient penalty with different gamma values."""
        # Create a small input that requires grad
        real_data = torch.randn(2, 3, 4, 4, requires_grad=True)
        real_outputs = torch.sum(real_data**2, dim=[1, 2, 3])  # Simple function to get gradient
        
        # Test with default gamma
        loss_fn_default = R1GradientPenalty()
        loss_default = loss_fn_default(real_data, real_outputs)
        assert isinstance(loss_default, torch.Tensor)
        
        # Test with custom gamma
        loss_fn_custom = R1GradientPenalty(gamma=5.0)
        loss_custom = loss_fn_custom(real_data, real_outputs)
        
        # Custom gamma should be half of default for same inputs
        assert abs(loss_custom.item() - loss_default.item() * 0.5) < 1e-5
    
    def test_zero_penalty_for_detached_input(self):
        """Test that zero penalty is returned if input doesn't require grad."""
        real_data = torch.randn(2, 3, 4, 4)  # No requires_grad=True
        real_outputs = torch.sum(real_data**2, dim=[1, 2, 3])
        
        loss_fn = R1GradientPenalty()
        # Since real_data doesn't require grad, gradient will be None 
        # and the penalty should be 0
        with pytest.warns(UserWarning, match="The .+ grad will be treated as zero"):
            loss = loss_fn(real_data, real_outputs)
            assert loss.item() == 0.0
    
    def test_none_gradient_handling(self):
        """Test handling of None gradients in R1GradientPenalty."""
        class MockModel(torch.nn.Module):
            def forward(self, x):
                # Return something unrelated to x to simulate None gradient
                return torch.ones(x.shape[0], 1)
                
        real_data = torch.randn(2, 3, 4, 4, requires_grad=True)
        model = MockModel()
        real_outputs = model(real_data)
        
        loss_fn = R1GradientPenalty()
        
        # When autograd.grad is called, it would normally return None for the gradient
        # of real_outputs with respect to real_data, but the function handles this case
        # by returning a zero tensor instead
        
        # Mock the autograd.grad to return [None]
        original_grad = torch.autograd.grad
        
        try:
            # Using a context manager to safely patch and restore the function
            class MockGrad:
                @staticmethod
                def apply(*args, **kwargs):
                    return [None]
                
            torch.autograd.grad = MockGrad.apply
            
            # This should not raise an error and should return a zero tensor
            loss = loss_fn(real_data, real_outputs)
            assert loss.item() == 0.0
            
        finally:
            # Restore the original function
            torch.autograd.grad = original_grad