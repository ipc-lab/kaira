"""Tests for the simplified datasets module."""

import numpy as np
import pytest
import torch

from kaira.data.datasets import (
    BinaryDataset,
    UniformDataset,
    GaussianDataset,
    CorrelatedDataset,
    FunctionDataset,
)


class TestBinaryDataset:
    """Test class for BinaryDataset."""

    def test_basic_initialization(self):
        """Test basic initialization and properties."""
        length = 10
        shape = (64,)
        prob = 0.3

        dataset = BinaryDataset(length=length, shape=shape, prob=prob, seed=42)

        assert len(dataset) == length

        # Test getting a sample
        sample = dataset[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == shape
        assert sample.dtype == torch.float32

        # Check that values are binary (0 or 1)
        assert torch.all((sample == 0) | (sample == 1))

    def test_multidimensional_shape(self):
        """Test with multidimensional shapes."""
        length = 5
        shape = (3, 32, 32)

        dataset = BinaryDataset(length=length, shape=shape, seed=42)

        sample = dataset[0]
        assert sample.shape == shape

    def test_shape_as_int(self):
        """Test with shape as single integer."""
        length = 5
        shape = 128

        dataset = BinaryDataset(length=length, shape=shape, seed=42)

        sample = dataset[0]
        assert sample.shape == (128,)

    def test_probability_control(self):
        """Test that the probability parameter controls the frequency of 1s."""
        length = 1000
        shape = (100,)
        prob = 0.3

        dataset = BinaryDataset(length=length, shape=shape, prob=prob, seed=42)

        # Get multiple samples and check overall frequency
        samples = torch.stack([dataset[i] for i in range(100)])
        actual_freq = samples.mean().item()

        # Should be approximately equal to prob (within some tolerance)
        assert abs(actual_freq - prob) < 0.1

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        length = 10
        shape = (5,)
        seed = 42

        dataset1 = BinaryDataset(length=length, shape=shape, seed=seed)
        dataset2 = BinaryDataset(length=length, shape=shape, seed=seed)

        sample1 = dataset1[0]
        sample2 = dataset2[0]

        assert torch.equal(sample1, sample2)


class TestUniformDataset:
    """Test class for UniformDataset."""

    def test_basic_initialization(self):
        """Test basic initialization and properties."""
        length = 10
        shape = (64,)
        low = -2.0
        high = 2.0

        dataset = UniformDataset(length=length, shape=shape, low=low, high=high, seed=42)

        assert len(dataset) == length

        # Test getting a sample
        sample = dataset[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == shape
        assert sample.dtype == torch.float32

        # Check that values are in the correct range
        assert torch.all(sample >= low)
        assert torch.all(sample <= high)

    def test_range_control(self):
        """Test that low and high parameters control the range."""
        length = 1000
        shape = (100,)
        low = -5.0
        high = 3.0

        dataset = UniformDataset(length=length, shape=shape, low=low, high=high, seed=42)

        # Get multiple samples and check range
        samples = torch.stack([dataset[i] for i in range(100)])
        
        assert torch.all(samples >= low)
        assert torch.all(samples <= high)
        
        # Check that we get values near both extremes
        assert samples.min().item() < low + 0.5
        assert samples.max().item() > high - 0.5


class TestGaussianDataset:
    """Test class for GaussianDataset."""

    def test_basic_initialization(self):
        """Test basic initialization and properties."""
        length = 10
        shape = (64,)
        mean = 1.0
        std = 2.0

        dataset = GaussianDataset(length=length, shape=shape, mean=mean, std=std, seed=42)

        assert len(dataset) == length

        # Test getting a sample
        sample = dataset[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == shape
        assert sample.dtype == torch.float32

    def test_statistical_properties(self):
        """Test that mean and std parameters are respected."""
        length = 10000
        shape = (100,)
        mean = 2.0
        std = 1.5

        dataset = GaussianDataset(length=length, shape=shape, mean=mean, std=std, seed=42)

        # Get many samples and check statistics
        samples = torch.stack([dataset[i] for i in range(100)])
        
        actual_mean = samples.mean().item()
        actual_std = samples.std().item()
        
        # Should be approximately equal (within some tolerance)
        assert abs(actual_mean - mean) < 0.2
        assert abs(actual_std - std) < 0.2


class TestCorrelatedDataset:
    """Test class for CorrelatedDataset."""

    def test_basic_initialization(self):
        """Test basic initialization and properties."""
        length = 10
        shape = (64,)
        correlation = 0.8

        dataset = CorrelatedDataset(length=length, shape=shape, correlation=correlation, seed=42)

        assert len(dataset) == length

        # Test getting a sample
        source, side_info = dataset[0]
        assert isinstance(source, torch.Tensor)
        assert isinstance(side_info, torch.Tensor)
        assert source.shape == shape
        assert side_info.shape == shape
        assert source.dtype == torch.float32
        assert side_info.dtype == torch.float32

    def test_correlation_control(self):
        """Test that correlation parameter controls the correlation."""
        length = 1000
        shape = (1000,)  # Larger shape for better correlation estimation
        target_correlation = 0.7

        dataset = CorrelatedDataset(length=length, shape=shape, correlation=target_correlation, seed=42)

        # Get a sample and check correlation
        source, side_info = dataset[0]
        
        # Calculate correlation coefficient
        correlation_matrix = torch.corrcoef(torch.stack([source.flatten(), side_info.flatten()]))
        actual_correlation = correlation_matrix[0, 1].item()
        
        # Should be approximately equal to target correlation (more relaxed tolerance)
        assert abs(actual_correlation - target_correlation) < 0.2


class TestFunctionDataset:
    """Test class for FunctionDataset."""

    def test_basic_initialization(self):
        """Test basic initialization and properties."""
        length = 10
        
        def generator_fn(idx):
            return torch.randn(5) * idx
        
        dataset = FunctionDataset(length=length, generator_fn=generator_fn, seed=42)

        assert len(dataset) == length

        # Test getting samples
        sample0 = dataset[0]
        sample1 = dataset[1]
        
        assert isinstance(sample0, torch.Tensor)
        assert isinstance(sample1, torch.Tensor)
        assert sample0.shape == (5,)
        assert sample1.shape == (5,)

    def test_custom_function(self):
        """Test with a custom generation function."""
        length = 5
        
        def sine_generator(idx):
            x = torch.linspace(0, 2*torch.pi, 100)
            return torch.sin(x + idx)
        
        dataset = FunctionDataset(length=length, generator_fn=sine_generator, seed=42)

        sample = dataset[0]
        assert sample.shape == (100,)
        
        # Check that different indices give different results
        sample0 = dataset[0]
        sample1 = dataset[1]
        assert not torch.equal(sample0, sample1)
