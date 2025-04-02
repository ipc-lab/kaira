import os
import random

import numpy as np
import torch

from kaira.utils import seed_everything


def test_seed_everything_reproducibility():
    """Test that seed_everything makes randomization reproducible."""
    # Set a specific seed
    seed_value = 42
    seed_everything(seed_value)

    # Get random numbers from different generators
    random_py = random.random()
    random_np = np.random.rand()
    random_torch = torch.rand(1).item()

    # Reset and seed again with the same value
    seed_everything(seed_value)

    # Check that we get the same values after re-seeding
    assert random_py == random.random()
    assert random_np == np.random.rand()
    assert random_torch == torch.rand(1).item()


def test_seed_everything_different_values():
    """Test that different seeds produce different random values."""
    # Seed with one value
    seed_everything(42)
    random_val_1 = torch.rand(10)

    # Seed with a different value
    seed_everything(123)
    random_val_2 = torch.rand(10)

    # Values should be different
    assert not torch.allclose(random_val_1, random_val_2)


def test_seed_everything_os_environ():
    """Test that seed_everything sets PYTHONHASHSEED environment variable."""
    seed_value = 42
    seed_everything(seed_value)

    assert os.environ["PYTHONHASHSEED"] == str(seed_value)


def test_seed_everything_cudnn_settings():
    """Test that seed_everything correctly sets CUDNN parameters."""
    # Test with default values
    seed_everything(42)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

    # Test with custom values
    seed_everything(42, cudnn_benchmark=True, cudnn_deterministic=False)
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True


def test_seed_everything_global_seed_state():
    """Test that seed_everything affects the global seed state."""
    # Set a specific seed
    seed_value = 42
    seed_everything(seed_value)
    
    # Generate some random values
    rand1 = torch.rand(5)
    rand2 = np.random.rand(5)
    rand3 = [random.random() for _ in range(5)]
    
    # Reset with the same seed
    seed_everything(seed_value)
    
    # Generate new random values - they should match the previous ones
    rand1_new = torch.rand(5)
    rand2_new = np.random.rand(5)
    rand3_new = [random.random() for _ in range(5)]
    
    # Check if the random values are identical
    assert torch.all(torch.eq(rand1, rand1_new))
    assert np.array_equal(rand2, rand2_new)
    assert rand3 == rand3_new
