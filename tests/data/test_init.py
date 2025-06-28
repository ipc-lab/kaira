"""Tests for the data module initialization."""

import pytest


class TestKairaDataInit:
    """Test class for kaira.data module initialization."""

    def test_imports_available(self):
        """Test that all expected classes and functions are importable."""
        from kaira.data import (
            BinaryDataset,
            UniformDataset,
            GaussianDataset,
            CorrelatedDataset,
            FunctionDataset,
            ImageDataset,
        )

        # Test that they're all classes/functions
        assert callable(BinaryDataset)
        assert callable(UniformDataset)
        assert callable(GaussianDataset)
        assert callable(CorrelatedDataset)
        assert callable(FunctionDataset)
        assert callable(ImageDataset)

    def test_all_exports_defined(self):
        """Test that __all__ contains all expected exports."""
        import kaira.data as data_module

        expected_exports = [
            "BinaryDataset",
            "UniformDataset",
            "GaussianDataset",
            "CorrelatedDataset",
            "FunctionDataset",
            "ImageDataset",
        ]

        assert hasattr(data_module, "__all__")
        assert set(data_module.__all__) == set(expected_exports)

    def test_module_docstring(self):
        """Test that the module has appropriate documentation."""
        import kaira.data as data_module

        assert data_module.__doc__ is not None
        assert "Data utilities for Kaira" in data_module.__doc__
        assert "memory-efficient" in data_module.__doc__

    def test_direct_class_instantiation(self):
        """Test that classes can be instantiated directly from the module."""
        from kaira.data import BinaryDataset, UniformDataset, GaussianDataset

        # Test basic instantiation
        binary_dataset = BinaryDataset(length=10, shape=(5,), seed=42)
        uniform_dataset = UniformDataset(length=10, shape=(5,), seed=42)
        gaussian_dataset = GaussianDataset(length=10, shape=(5,), seed=42)

        assert len(binary_dataset) == 10
        assert len(uniform_dataset) == 10
        assert len(gaussian_dataset) == 10

    def test_module_structure(self):
        """Test that the module has the expected structure."""
        import kaira.data

        # Check that submodules are accessible
        assert hasattr(kaira.data, 'datasets')
        assert hasattr(kaira.data, 'sample_data')

    def test_no_old_classes(self):
        """Test that old classes are no longer available."""
        import kaira.data

        # These should not be available anymore
        old_classes = [
            'BinaryTensorDataset',
            'UniformTensorDataset', 
            'WynerZivCorrelationDataset',
            'SampleImagesDataset',
            'TorchVisionDataset',
        ]

        for old_class in old_classes:
            assert not hasattr(kaira.data, old_class)
