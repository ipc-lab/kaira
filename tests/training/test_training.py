"""Comprehensive tests for kaira.training module.

This module provides 100% test coverage for:
- TrainingArguments class and all its methods
- Trainer class and all its methods
- TrainingArgumentsMixin class
- _extract_config_value function
- All error handling and edge cases
"""

import tempfile
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments as HFTrainingArguments

from kaira.training import Trainer, TrainingArguments
from kaira.training.arguments import TrainingArgumentsMixin, _extract_config_value


class MockModel(nn.Module):
    """Mock model for testing trainer functionality."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, **inputs):
        # Simple mock forward pass
        return torch.randn(2, 5)


class TestExtractConfigValue:
    """Test the _extract_config_value utility function."""

    def test_extract_from_object_with_attribute(self):
        """Test extracting value from object with attribute."""
        obj = Mock()
        obj.test_key = "test_value"
        result = _extract_config_value(obj, "test_key", "default")
        assert result == "test_value"

    def test_extract_from_dict(self):
        """Test extracting value from dictionary."""
        config = {"test_key": "test_value"}
        result = _extract_config_value(config, "test_key", "default")
        assert result == "test_value"

    def test_extract_from_dict_missing_key(self):
        """Test extracting missing key from dictionary returns default."""
        config = {"other_key": "other_value"}
        result = _extract_config_value(config, "test_key", "default")
        assert result == "default"

    def test_extract_from_dictconfig(self):
        """Test extracting value from OmegaConf DictConfig."""
        config = OmegaConf.create({"test_key": "test_value"})
        result = _extract_config_value(config, "test_key", "default")
        assert result == "test_value"

    def test_extract_from_dictconfig_nested(self):
        """Test extracting nested value from DictConfig."""
        config = OmegaConf.create({"nested": {"test_key": "test_value"}})
        result = _extract_config_value(config, "nested.test_key", "default")
        assert result == "test_value"

    def test_extract_from_dictconfig_missing_key(self):
        """Test extracting missing key from DictConfig returns default."""
        config = OmegaConf.create({"other_key": "other_value"})
        result = _extract_config_value(config, "test_key", "default")
        assert result == "default"

    def test_extract_from_other_object(self):
        """Test extracting from object without attribute returns default."""
        obj = "simple_string"
        result = _extract_config_value(obj, "test_key", "default")
        assert result == "default"

    def test_extract_with_none_default(self):
        """Test extracting with None as default."""
        config = {}
        result = _extract_config_value(config, "missing_key")
        assert result is None


class TestTrainingArgumentsMixin:
    """Test the TrainingArgumentsMixin class."""

    class MockTrainingArguments(TrainingArgumentsMixin):
        """Mock class that uses TrainingArgumentsMixin."""

        def __init__(self, output_dir="./results", learning_rate=1e-4, **kwargs):
            self.output_dir = output_dir
            self.learning_rate = learning_rate
            for k, v in kwargs.items():
                setattr(self, k, v)

        @classmethod
        def _get_valid_parameters(cls):
            return {"output_dir", "learning_rate", "num_train_epochs", "per_device_train_batch_size"}

    def test_from_hydra_with_dictconfig(self):
        """Test creating instance from DictConfig."""
        config = OmegaConf.create({"output_dir": "./test_results", "learning_rate": 2e-4, "num_train_epochs": 5})

        instance = self.MockTrainingArguments.from_hydra(config)
        assert instance.output_dir == "./test_results"
        assert instance.learning_rate == 2e-4
        assert instance.num_train_epochs == 5

    def test_from_hydra_with_dict(self):
        """Test creating instance from plain dict."""
        config = {"output_dir": "./test_results", "learning_rate": 2e-4}

        instance = self.MockTrainingArguments.from_hydra(config)
        assert instance.output_dir == "./test_results"
        assert instance.learning_rate == 2e-4

    def test_from_hydra_with_nested_training(self):
        """Test creating instance from config with nested training section."""
        config = OmegaConf.create({"training": {"output_dir": "./nested_results", "learning_rate": 3e-4}, "model": {"hidden_size": 128}})

        instance = self.MockTrainingArguments.from_hydra(config)
        assert instance.output_dir == "./nested_results"
        assert instance.learning_rate == 3e-4

    def test_from_hydra_with_override_kwargs(self):
        """Test creating instance with override kwargs."""
        config = {"output_dir": "./test_results", "learning_rate": 2e-4}

        instance = self.MockTrainingArguments.from_hydra(config, learning_rate=5e-4, num_train_epochs=10)
        assert instance.output_dir == "./test_results"
        assert instance.learning_rate == 5e-4  # Overridden
        assert instance.num_train_epochs == 10

    def test_from_hydra_filters_invalid_parameters(self):
        """Test that invalid parameters are filtered out."""
        config = {"output_dir": "./test_results", "learning_rate": 2e-4, "invalid_param": "should_be_filtered"}

        instance = self.MockTrainingArguments.from_hydra(config)
        assert instance.output_dir == "./test_results"
        assert instance.learning_rate == 2e-4
        assert not hasattr(instance, "invalid_param")

    def test_from_hydra_invalid_type(self):
        """Test that invalid config type raises ValueError."""
        with pytest.raises(ValueError, match="Expected DictConfig or dict"):
            self.MockTrainingArguments.from_hydra("invalid_config")

    def test_from_dict(self):
        """Test creating instance from dict."""
        config = {"output_dir": "./dict_results", "learning_rate": 1e-3, "num_train_epochs": 15}

        instance = self.MockTrainingArguments.from_dict(config)
        assert instance.output_dir == "./dict_results"
        assert instance.learning_rate == 1e-3
        assert instance.num_train_epochs == 15

    def test_from_dict_with_override_kwargs(self):
        """Test from_dict with override kwargs."""
        config = {"output_dir": "./dict_results", "learning_rate": 1e-3}

        instance = self.MockTrainingArguments.from_dict(config, learning_rate=2e-3, per_device_train_batch_size=16)
        assert instance.output_dir == "./dict_results"
        assert instance.learning_rate == 2e-3  # Overridden
        assert instance.per_device_train_batch_size == 16

    def test_from_dict_filters_invalid_parameters(self):
        """Test that from_dict filters invalid parameters."""
        config = {"output_dir": "./dict_results", "learning_rate": 1e-3, "invalid_param": "filtered"}

        instance = self.MockTrainingArguments.from_dict(config)
        assert instance.output_dir == "./dict_results"
        assert instance.learning_rate == 1e-3
        assert not hasattr(instance, "invalid_param")

    def test_get_valid_parameters_with_mixin(self):
        """Test _get_valid_parameters works correctly with mixin."""
        params = self.MockTrainingArguments._get_valid_parameters()

        # Should include expected parameters
        assert "output_dir" in params
        assert "learning_rate" in params
        assert "num_train_epochs" in params
        assert "per_device_train_batch_size" in params

        # Should not include internal parameters
        assert "self" not in params


class TestTrainingArguments:
    """Test the TrainingArguments class."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            args = TrainingArguments(output_dir=temp_dir)

            # Check default communication parameters
            assert args.snr_min == 0.0
            assert args.snr_max == 20.0
            assert args.noise_variance_min == 0.1
            assert args.noise_variance_max == 2.0
            assert args.channel_uses is None
            assert args.code_length is None
            assert args.info_length is None
            assert args.channel_type == "awgn"

            # Check training parameters
            assert args.output_dir == temp_dir
            assert args.num_train_epochs == 10.0
            assert args.per_device_train_batch_size == 32
            assert args.learning_rate == 1e-4

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            args = TrainingArguments(output_dir=temp_dir, snr_min=5.0, snr_max=25.0, noise_variance_min=0.05, noise_variance_max=3.0, channel_uses=128, code_length=256, info_length=64, channel_type="rayleigh", num_train_epochs=20.0, per_device_train_batch_size=64, learning_rate=2e-4)

            assert args.snr_min == 5.0
            assert args.snr_max == 25.0
            assert args.noise_variance_min == 0.05
            assert args.noise_variance_max == 3.0
            assert args.channel_uses == 128
            assert args.code_length == 256
            assert args.info_length == 64
            assert args.channel_type == "rayleigh"
            assert args.num_train_epochs == 20.0
            assert args.per_device_train_batch_size == 64
            assert args.learning_rate == 2e-4

    def test_from_training_arguments(self):
        """Test creating TrainingArguments from HF TrainingArguments."""
        with tempfile.TemporaryDirectory() as temp_dir:
            hf_args = HFTrainingArguments(output_dir=temp_dir, num_train_epochs=5, per_device_train_batch_size=16, learning_rate=5e-4)

            args = TrainingArguments.from_training_arguments(hf_args)

            assert args.output_dir == temp_dir
            assert args.num_train_epochs == 5
            assert args.per_device_train_batch_size == 16
            assert args.learning_rate == 5e-4
            # Communication parameters should have defaults
            assert args.snr_min == 0.0
            assert args.snr_max == 20.0

    def test_from_training_arguments_with_overrides(self):
        """Test from_training_arguments with override kwargs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            hf_args = HFTrainingArguments(output_dir=temp_dir, num_train_epochs=5, learning_rate=5e-4)

            args = TrainingArguments.from_training_arguments(hf_args, snr_min=10.0, snr_max=30.0, learning_rate=1e-3)  # Override

            assert args.output_dir == temp_dir
            assert args.num_train_epochs == 5
            assert args.learning_rate == 1e-3  # Overridden
            assert args.snr_min == 10.0
            assert args.snr_max == 30.0

    def test_from_hydra_config(self):
        """Test creating TrainingArguments from Hydra config."""
        hydra_cfg = OmegaConf.create({"training": {"output_dir": "./hydra_results", "num_train_epochs": 8, "learning_rate": 3e-4, "snr_min": 2.0, "snr_max": 18.0}})

        args = TrainingArguments.from_hydra_config(hydra_cfg)

        assert args.output_dir == "./hydra_results"
        assert args.num_train_epochs == 8
        assert args.learning_rate == 3e-4
        assert args.snr_min == 2.0
        assert args.snr_max == 18.0

    def test_from_hydra_config_with_overrides(self):
        """Test from_hydra_config with override kwargs."""
        hydra_cfg = OmegaConf.create({"training": {"output_dir": "./hydra_results", "learning_rate": 3e-4}})

        args = TrainingArguments.from_hydra_config(hydra_cfg, learning_rate=6e-4, channel_uses=64)  # Override

        assert args.output_dir == "./hydra_results"
        assert args.learning_rate == 6e-4  # Overridden
        assert args.channel_uses == 64

    def test_convert_to_training_arguments_with_training_arguments(self):
        """Test _convert_to_training_arguments with TrainingArguments input."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_args = TrainingArguments(output_dir=temp_dir)
            converted_args = TrainingArguments._convert_to_training_arguments(original_args)

            assert converted_args is original_args  # Should return same instance

    def test_convert_to_training_arguments_with_dictconfig(self):
        """Test _convert_to_training_arguments with DictConfig."""
        config = OmegaConf.create({"output_dir": "./convert_results", "num_train_epochs": 12, "learning_rate": 7e-4})

        args = TrainingArguments._convert_to_training_arguments(config)

        assert args.output_dir == "./convert_results"
        assert args.num_train_epochs == 12
        assert args.learning_rate == 7e-4

    def test_convert_to_training_arguments_with_dict(self):
        """Test _convert_to_training_arguments with dict."""
        config = {"output_dir": "./convert_dict_results", "num_train_epochs": 15, "per_device_train_batch_size": 8}

        args = TrainingArguments._convert_to_training_arguments(config)

        assert args.output_dir == "./convert_dict_results"
        assert args.num_train_epochs == 15
        assert args.per_device_train_batch_size == 8

    def test_convert_to_training_arguments_with_object_with_vars(self):
        """Test _convert_to_training_arguments with object that has vars()."""

        class MockConfig:
            def __init__(self):
                self.output_dir = "./object_results"
                self.learning_rate = 9e-4

        config = MockConfig()
        args = TrainingArguments._convert_to_training_arguments(config)

        assert args.output_dir == "./object_results"
        assert args.learning_rate == 9e-4

    def test_convert_to_training_arguments_with_invalid_object(self):
        """Test _convert_to_training_arguments with object that can't be converted."""
        # Create an object that can't be converted with vars()
        # Use a built-in type that doesn't have __dict__
        invalid_config = 42  # Integer can't use vars()

        with pytest.raises(ValueError, match="Cannot convert args of type"):
            TrainingArguments._convert_to_training_arguments(invalid_config)

    def test_to_dict(self):
        """Test converting TrainingArguments to dict."""
        with tempfile.TemporaryDirectory() as temp_dir:
            args = TrainingArguments(output_dir=temp_dir, snr_min=1.0, snr_max=15.0, channel_uses=32, num_train_epochs=7)

            result_dict = args.to_dict()

            # Check that communication parameters are included
            assert result_dict["snr_min"] == 1.0
            assert result_dict["snr_max"] == 15.0
            assert result_dict["channel_uses"] == 32

            # Check that training parameters are included
            assert result_dict["output_dir"] == temp_dir
            assert result_dict["num_train_epochs"] == 7

    def test_to_hydra_config(self):
        """Test converting TrainingArguments to Hydra config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            args = TrainingArguments(output_dir=temp_dir, snr_min=3.0, channel_type="fading")

            hydra_config = args.to_hydra_config()

            assert isinstance(hydra_config, DictConfig)
            assert hydra_config.output_dir == temp_dir
            assert hydra_config.snr_min == 3.0
            assert hydra_config.channel_type == "fading"

    def test_get_snr_range(self):
        """Test get_snr_range method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            args = TrainingArguments(output_dir=temp_dir, snr_min=5.0, snr_max=25.0)

            snr_range = args.get_snr_range()
            assert snr_range == (5.0, 25.0)

    def test_get_noise_variance_range(self):
        """Test get_noise_variance_range method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            args = TrainingArguments(output_dir=temp_dir, noise_variance_min=0.05, noise_variance_max=2.5)

            noise_range = args.get_noise_variance_range()
            assert noise_range == (0.05, 2.5)

    def test_update_from_hydra_with_dictconfig(self):
        """Test update_from_hydra with DictConfig."""
        with tempfile.TemporaryDirectory() as temp_dir:
            args = TrainingArguments(output_dir=temp_dir)

            hydra_config = OmegaConf.create({"snr_min": 8.0, "snr_max": 22.0, "learning_rate": 4e-4})

            args.update_from_hydra(hydra_config)

            assert args.snr_min == 8.0
            assert args.snr_max == 22.0
            assert args.learning_rate == 4e-4

    def test_update_from_hydra_with_dict(self):
        """Test update_from_hydra with plain dict."""
        with tempfile.TemporaryDirectory() as temp_dir:
            args = TrainingArguments(output_dir=temp_dir)

            config = {"channel_uses": 96, "num_train_epochs": 25}

            args.update_from_hydra(config)

            assert args.channel_uses == 96
            assert args.num_train_epochs == 25

    def test_update_from_hydra_with_nested_training(self):
        """Test update_from_hydra with nested training config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            args = TrainingArguments(output_dir=temp_dir)

            config = {"training": {"code_length": 512, "info_length": 128}, "model": {"hidden_size": 256}}

            args.update_from_hydra(config)

            assert args.code_length == 512
            assert args.info_length == 128
            # Model config should not affect args

    def test_get_valid_parameters(self):
        """Test _get_valid_parameters method."""
        params = TrainingArguments._get_valid_parameters()

        # Should include both HF parameters and custom communication parameters
        assert "output_dir" in params
        assert "num_train_epochs" in params
        assert "learning_rate" in params
        assert "snr_min" in params
        assert "snr_max" in params
        assert "channel_uses" in params


class TestTrainer:
    """Test the Trainer class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        return MockModel()

    @pytest.fixture
    def training_args(self):
        """Create TrainingArguments for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            return TrainingArguments(output_dir=temp_dir, num_train_epochs=1, per_device_train_batch_size=2, per_device_eval_batch_size=2, logging_steps=1, eval_steps=1, save_steps=1, eval_strategy="no")  # Disable evaluation to avoid needing eval_dataset

    def test_init_with_training_arguments(self, mock_model, training_args):
        """Test Trainer initialization with TrainingArguments."""
        trainer = Trainer(model=mock_model, args=training_args)

        assert trainer.model is mock_model
        assert isinstance(trainer.args, TrainingArguments)
        assert trainer.args.output_dir == training_args.output_dir

    def test_init_with_hf_training_arguments(self, mock_model):
        """Test Trainer initialization with HF TrainingArguments."""
        with tempfile.TemporaryDirectory() as temp_dir:
            hf_args = HFTrainingArguments(output_dir=temp_dir, num_train_epochs=2, per_device_train_batch_size=4)

            trainer = Trainer(model=mock_model, args=hf_args)

            assert trainer.model is mock_model
            assert isinstance(trainer.args, TrainingArguments)
            assert trainer.args.output_dir == temp_dir
            assert trainer.args.num_train_epochs == 2
            assert trainer.args.per_device_train_batch_size == 4

    def test_init_with_dictconfig(self, mock_model):
        """Test Trainer initialization with DictConfig."""
        config = OmegaConf.create({"output_dir": "./dictconfig_results", "num_train_epochs": 3, "learning_rate": 2e-4, "eval_strategy": "no"})

        trainer = Trainer(model=mock_model, args=config)

        assert trainer.model is mock_model
        assert isinstance(trainer.args, TrainingArguments)
        assert trainer.args.output_dir == "./dictconfig_results"
        assert trainer.args.num_train_epochs == 3
        assert trainer.args.learning_rate == 2e-4

    def test_init_with_dict(self, mock_model):
        """Test Trainer initialization with plain dict."""
        config = {"output_dir": "./dict_results", "num_train_epochs": 4, "per_device_train_batch_size": 8, "eval_strategy": "no"}

        trainer = Trainer(model=mock_model, args=config)

        assert trainer.model is mock_model
        assert isinstance(trainer.args, TrainingArguments)
        assert trainer.args.output_dir == "./dict_results"
        assert trainer.args.num_train_epochs == 4
        assert trainer.args.per_device_train_batch_size == 8

    def test_init_with_other_object(self, mock_model):
        """Test Trainer initialization with other object type."""

        class MockConfig:
            def __init__(self):
                self.output_dir = "./other_results"
                self.learning_rate = 1e-3
                self.eval_strategy = "no"

        config = MockConfig()
        trainer = Trainer(model=mock_model, args=config)

        assert trainer.model is mock_model
        assert isinstance(trainer.args, TrainingArguments)
        assert trainer.args.output_dir == "./other_results"
        assert trainer.args.learning_rate == 1e-3

    def test_init_with_kwargs(self, mock_model, training_args):
        """Test Trainer initialization with additional kwargs."""
        mock_tokenizer = Mock()
        mock_compute_metrics = Mock()

        trainer = Trainer(model=mock_model, args=training_args, processing_class=mock_tokenizer, compute_metrics=mock_compute_metrics)  # Use processing_class instead of tokenizer

        assert trainer.model is mock_model
        assert trainer.processing_class is mock_tokenizer
        assert trainer.compute_metrics is mock_compute_metrics

    def test_from_hydra_config(self, mock_model):
        """Test creating Trainer from Hydra configuration."""
        hydra_cfg = OmegaConf.create({"training": {"output_dir": "./hydra_trainer_results", "num_train_epochs": 6, "learning_rate": 3e-4, "eval_strategy": "no"}})

        trainer = Trainer.from_hydra_config(hydra_cfg, mock_model)

        assert trainer.model is mock_model
        assert isinstance(trainer.args, TrainingArguments)
        assert trainer.args.output_dir == "./hydra_trainer_results"
        assert trainer.args.num_train_epochs == 6
        assert trainer.args.learning_rate == 3e-4

    def test_from_hydra_config_with_kwargs(self, mock_model):
        """Test from_hydra_config with additional kwargs."""
        hydra_cfg = OmegaConf.create({"training": {"output_dir": "./hydra_kwargs_results", "eval_strategy": "no"}})

        mock_tokenizer = Mock()

        trainer = Trainer.from_hydra_config(hydra_cfg, mock_model, processing_class=mock_tokenizer)  # Use processing_class instead of tokenizer

        assert trainer.model is mock_model
        assert trainer.processing_class is mock_tokenizer
        assert trainer.args.output_dir == "./hydra_kwargs_results"

    def test_from_training_args(self, mock_model):
        """Test creating Trainer from HF TrainingArguments."""
        with tempfile.TemporaryDirectory() as temp_dir:
            hf_args = HFTrainingArguments(output_dir=temp_dir, num_train_epochs=7, learning_rate=5e-4)

            trainer = Trainer.from_training_args(hf_args, mock_model)

            assert trainer.model is mock_model
            assert isinstance(trainer.args, TrainingArguments)
            assert trainer.args.output_dir == temp_dir
            assert trainer.args.num_train_epochs == 7
            assert trainer.args.learning_rate == 5e-4

    def test_from_training_args_with_kwargs(self, mock_model):
        """Test from_training_args with additional kwargs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            hf_args = HFTrainingArguments(output_dir=temp_dir)
            mock_data_collator = Mock()

            trainer = Trainer.from_training_args(hf_args, mock_model, data_collator=mock_data_collator)

            assert trainer.model is mock_model
            assert trainer.data_collator is mock_data_collator

    def test_from_training_arguments_with_attribute_errors(self):
        """Test from_training_arguments handles attribute access errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create regular HF args and test normal flow
            hf_args = HFTrainingArguments(output_dir=temp_dir)

            # Test with an object that has problematic attributes
            class ProblematicArgs:
                def __init__(self):
                    # Set basic required attributes directly
                    for attr_name in dir(hf_args):
                        if not attr_name.startswith("_") and hasattr(hf_args, attr_name):
                            try:
                                setattr(self, attr_name, getattr(hf_args, attr_name))
                            except (AttributeError, TypeError, ValueError):
                                # Specifically handle expected attribute access errors
                                # Some attributes may be properties or methods that can't be copied
                                continue

                    # Add problematic attribute that will cause issues
                    self._problematic = "problem"

                def __dir__(self):
                    # Include an attribute that will cause problems when accessed
                    base_attrs = [attr for attr in super().__dir__() if not attr.startswith("_")]
                    return base_attrs + ["error_prone_attr"]

                def __getattribute__(self, name):
                    if name == "error_prone_attr":
                        raise TypeError("Cannot access this attribute")
                    return super().__getattribute__(name)

            problematic_args = ProblematicArgs()

            # This should handle the TypeError gracefully and still create valid args
            args = TrainingArguments.from_training_arguments(problematic_args)

            assert args.output_dir == temp_dir
            assert hasattr(args, "num_train_epochs")


class TestTrainingModuleIntegration:
    """Test integration between different components of the training module."""

    def test_training_arguments_with_all_parameters(self):
        """Test TrainingArguments with all communication and training parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            args = TrainingArguments(
                # Communication parameters
                snr_min=1.0,
                snr_max=30.0,
                noise_variance_min=0.01,
                noise_variance_max=5.0,
                channel_uses=256,
                code_length=1024,
                info_length=512,
                channel_type="rayleigh",
                # Training parameters
                output_dir=temp_dir,
                num_train_epochs=50,
                per_device_train_batch_size=128,
                per_device_eval_batch_size=64,
                learning_rate=1e-3,
                warmup_steps=2000,
                logging_steps=50,
                eval_steps=250,
                save_steps=500,
                eval_strategy="steps",
                logging_strategy="steps",
                save_strategy="steps",
            )

            # Test all parameters are set correctly
            assert args.snr_min == 1.0
            assert args.snr_max == 30.0
            assert args.noise_variance_min == 0.01
            assert args.noise_variance_max == 5.0
            assert args.channel_uses == 256
            assert args.code_length == 1024
            assert args.info_length == 512
            assert args.channel_type == "rayleigh"
            assert args.output_dir == temp_dir
            assert args.num_train_epochs == 50
            assert args.per_device_train_batch_size == 128
            assert args.per_device_eval_batch_size == 64
            assert args.learning_rate == 1e-3
            assert args.warmup_steps == 2000
            assert args.logging_steps == 50
            assert args.eval_steps == 250
            assert args.save_steps == 500

    def test_full_workflow_with_hydra_config(self):
        """Test complete workflow from Hydra config to Trainer."""
        hydra_cfg = OmegaConf.create(
            {"training": {"output_dir": "./full_workflow_results", "num_train_epochs": 10, "per_device_train_batch_size": 32, "learning_rate": 2e-4, "snr_min": 5.0, "snr_max": 25.0, "channel_uses": 128, "channel_type": "awgn", "eval_strategy": "no"}, "model": {"hidden_size": 256}}
        )

        # Create TrainingArguments from Hydra config
        args = TrainingArguments.from_hydra_config(hydra_cfg)

        # Verify args are correct
        assert args.output_dir == "./full_workflow_results"
        assert args.num_train_epochs == 10
        assert args.per_device_train_batch_size == 32
        assert args.learning_rate == 2e-4
        assert args.snr_min == 5.0
        assert args.snr_max == 25.0
        assert args.channel_uses == 128
        assert args.channel_type == "awgn"

        # Create Trainer with the args
        mock_model = MockModel()
        trainer = Trainer(model=mock_model, args=args)

        assert trainer.model is mock_model
        assert isinstance(trainer.args, TrainingArguments)
        assert trainer.args.output_dir == "./full_workflow_results"

        # Test creating trainer directly from hydra config
        trainer2 = Trainer.from_hydra_config(hydra_cfg, mock_model)
        assert trainer2.args.output_dir == "./full_workflow_results"
        assert trainer2.args.snr_min == 5.0

    def test_conversion_between_argument_types(self):
        """Test conversion between different argument types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Start with HF TrainingArguments
            hf_args = HFTrainingArguments(output_dir=temp_dir, num_train_epochs=8, learning_rate=3e-4)

            # Convert to Kaira TrainingArguments
            kaira_args = TrainingArguments.from_training_arguments(hf_args, snr_min=2.0, channel_uses=64)

            # Verify conversion
            assert kaira_args.output_dir == temp_dir
            assert kaira_args.num_train_epochs == 8
            assert kaira_args.learning_rate == 3e-4
            assert kaira_args.snr_min == 2.0
            assert kaira_args.channel_uses == 64

            # Convert to dict
            args_dict = kaira_args.to_dict()
            assert args_dict["output_dir"] == temp_dir
            assert args_dict["snr_min"] == 2.0
            assert args_dict["channel_uses"] == 64

            # Convert to Hydra config
            hydra_config = kaira_args.to_hydra_config()
            assert hydra_config.output_dir == temp_dir
            assert hydra_config.snr_min == 2.0
            assert hydra_config.channel_uses == 64

            # Create new args from dict
            new_args = TrainingArguments.from_dict(args_dict)
            assert new_args.output_dir == temp_dir
            assert new_args.snr_min == 2.0
            assert new_args.channel_uses == 64


class TestTrainingModuleImports:
    """Test that all public classes and functions are properly exported."""

    def test_training_module_exports(self):
        """Test that training module exports expected classes."""
        from kaira.training import Trainer, TrainingArguments

        # Verify classes are importable and are the expected types
        assert TrainingArguments is not None
        assert Trainer is not None

        # Test that they are the correct classes
        assert TrainingArguments.__name__ == "TrainingArguments"
        assert Trainer.__name__ == "Trainer"

    def test_top_level_imports(self):
        """Test that classes are available from top-level kaira import."""
        from kaira import Trainer, TrainingArguments

        assert TrainingArguments is not None
        assert Trainer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
