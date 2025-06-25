"""Tests for the kaira.data module initialization."""


class TestKairaDataInit:
    """Test class for kaira.data module initialization and imports."""

    def test_imports_available(self):
        """Test that all expected classes and functions are importable."""
        from kaira.data import (
            BinaryTensorDataset,
            SampleImagesDataset,
            TorchVisionDataset,
            UniformTensorDataset,
            WynerZivCorrelationDataset,
            download_image,
        )

        # Verify all imports are callable/classes
        assert callable(BinaryTensorDataset)
        assert callable(UniformTensorDataset)
        assert callable(WynerZivCorrelationDataset)
        assert callable(SampleImagesDataset)
        assert callable(TorchVisionDataset)
        assert callable(download_image)

    def test_all_exports_defined(self):
        """Test that __all__ contains all expected exports."""
        import kaira.data as data_module

        expected_exports = [
            "BinaryTensorDataset",
            "UniformTensorDataset",
            "WynerZivCorrelationDataset",
            "SampleImagesDataset",
            "TorchVisionDataset",
            "download_image",
        ]

        assert hasattr(data_module, "__all__")
        assert set(data_module.__all__) == set(expected_exports)

    def test_module_docstring(self):
        """Test that the module has appropriate documentation."""
        import kaira.data as data_module

        assert data_module.__doc__ is not None
        assert "Data utilities for Kaira" in data_module.__doc__
        assert "HuggingFace datasets" in data_module.__doc__

    def test_direct_class_instantiation(self):
        """Test that classes can be instantiated directly from the module."""
        from kaira.data import BinaryTensorDataset, UniformTensorDataset, WynerZivCorrelationDataset

        # Test basic instantiation
        binary_dataset = BinaryTensorDataset(n_samples=1, feature_shape=(10,))
        assert len(binary_dataset) == 1

        uniform_dataset = UniformTensorDataset(n_samples=1, feature_shape=(10,))
        assert len(uniform_dataset) == 1

        wyner_ziv_dataset = WynerZivCorrelationDataset(n_samples=1, feature_shape=(10,), correlation_type="binary")
        assert len(wyner_ziv_dataset) == 1
