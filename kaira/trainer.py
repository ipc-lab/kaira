"""Unified trainer for communication models using Transformers framework.

This module provides a flexible trainer that supports multiple configuration systems
for training arguments, while models handle their own configuration separately.

The trainer works with BaseModel instances and lets the models handle their own
channel simulation and constraints internally.

Examples:
    Using models with Hugging Face configurations:
    >>> from kaira import KairaTrainer
    >>> from kaira.models import BaseModel, ModelConfig
    >>> from transformers import TrainingArguments
    >>>
    >>> # Configure the model
    >>> model_config = ModelConfig(input_dim=512, channel_uses=64)
    >>> model = BaseModel.from_pretrained_config(model_config)
    >>>
    >>> # Configure training
    >>> training_args = TrainingArguments(output_dir="./results", num_train_epochs=10)
    >>> trainer = KairaTrainer(model, training_args)

    Using Hydra configurations:
    >>> # Model handles its own configuration
    >>> model = BaseModel.from_hydra_config(hydra_cfg.model)
    >>> trainer = KairaTrainer.from_hydra_config(hydra_cfg, model)

    Using plain dictionaries:
    >>> # Model handles configuration internally
    >>> model = BaseModel.from_config({"input_dim": 512, "channel_uses": 64})
    >>> # Training config
    >>> training_config = {"output_dir": "./results", "num_train_epochs": 10}
    >>> trainer = KairaTrainer(model, training_config)
"""

from typing import Any, Dict, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import Trainer
from transformers import TrainingArguments as HFTrainingArguments

from kaira.losses import LossRegistry


def _extract_config_value(config: Any, key: str, default: Any = None) -> Any:
    """Extract value from various config types (PretrainedConfig, TrainingArguments, DictConfig,
    dict)."""
    if hasattr(config, key):
        return getattr(config, key)
    elif isinstance(config, dict):
        return config.get(key, default)
    elif isinstance(config, DictConfig):
        return OmegaConf.select(config, key, default=default)
    else:
        return default


class TrainingArgumentsMixin:
    """Mixin providing Hydra and dict-based constructors for TrainingArguments subclasses."""

    @classmethod
    def from_hydra(cls, hydra_config: Union[Any, dict], **override_kwargs):
        """Create instance from Hydra configuration (DictConfig or dict)."""
        if isinstance(hydra_config, DictConfig):
            config_dict = OmegaConf.to_container(hydra_config, resolve=True)
        elif isinstance(hydra_config, dict):
            config_dict = hydra_config.copy()
        else:
            raise ValueError(f"Expected DictConfig or dict, got {type(hydra_config)}")

        # If nested, extract training segment
        if "training" in config_dict:
            config_segment = config_dict["training"]
        else:
            config_segment = config_dict

        config_segment.update(override_kwargs)
        # Filter parameters
        valid = cls._get_valid_parameters()
        filtered = {k: v for k, v in config_segment.items() if k in valid}
        return cls(**filtered)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **override_kwargs):
        """Create instance from plain dict."""
        merged = config_dict.copy()
        merged.update(override_kwargs)
        valid = cls._get_valid_parameters()
        filtered = {k: v for k, v in merged.items() if k in valid}
        return cls(**filtered)

    @classmethod
    def _get_valid_parameters(cls) -> set:
        """Get set of valid parameter names for this class."""
        # Get parameters from the class __init__ method
        import inspect

        init_signature = inspect.signature(cls.__init__)
        return set(init_signature.parameters.keys())


class TrainingArguments(TrainingArgumentsMixin, HFTrainingArguments):
    """Flexible training arguments that support both Hydra configs and TrainingArguments.

    This class extends transformers.TrainingArguments to provide seamless integration
    with Hydra configuration management while maintaining full compatibility with
    Hugging Face ecosystem. It supports:

    - Direct instantiation from Hydra DictConfig
    - Conversion from/to standard TrainingArguments
    - Communication-specific parameters
    - Automatic parameter filtering and validation

    Examples:
        >>> # From Hydra config
        >>> hydra_config = OmegaConf.create({"output_dir": "./results", "num_train_epochs": 10})
        >>> args = TrainingArguments.from_hydra(hydra_config)

        >>> # From TrainingArguments
        >>> training_args = TrainingArguments(output_dir="./results")
        >>> args = TrainingArguments.from_training_arguments(training_args)

        >>> # With communication parameters
        >>> args = TrainingArguments(
        ...     output_dir="./results",
        ...     snr_min=0.0,
        ...     snr_max=20.0,
        ...     channel_uses=64
        ... )
    """

    def __init__(
        self,
        # Communication-specific parameters
        snr_min: float = 0.0,
        snr_max: float = 20.0,
        noise_variance_min: float = 0.1,
        noise_variance_max: float = 2.0,
        channel_uses: Optional[int] = None,
        code_length: Optional[int] = None,
        info_length: Optional[int] = None,
        channel_type: str = "awgn",
        # Training parameters with defaults that work well for communication models
        output_dir: str = "./results",
        num_train_epochs: float = 10.0,
        per_device_train_batch_size: int = 32,
        per_device_eval_batch_size: int = 32,
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        logging_steps: int = 100,
        eval_steps: int = 500,
        save_steps: int = 1000,
        evaluation_strategy: str = "steps",
        logging_strategy: str = "steps",
        save_strategy: str = "steps",
        **kwargs,
    ):
        """Initialize TrainingArguments.

        Args:
            snr_min: Minimum SNR value for training
            snr_max: Maximum SNR value for training
            noise_variance_min: Minimum noise variance
            noise_variance_max: Maximum noise variance
            channel_uses: Number of channel uses
            code_length: Length of the code
            info_length: Length of information bits
            channel_type: Type of channel simulation
            output_dir: Output directory for results
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Training batch size per device
            per_device_eval_batch_size: Evaluation batch size per device
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            logging_steps: Log every X steps
            eval_steps: Evaluate every X steps
            save_steps: Save every X steps
            evaluation_strategy: Evaluation strategy
            logging_strategy: Logging strategy
            save_strategy: Save strategy
            **kwargs: Additional arguments passed to TrainingArguments
        """
        # Initialize parent class with filtered kwargs
        super().__init__(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            evaluation_strategy=evaluation_strategy,
            logging_strategy=logging_strategy,
            save_strategy=save_strategy,
            **kwargs,
        )

        # Store communication-specific parameters
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.noise_variance_min = noise_variance_min
        self.noise_variance_max = noise_variance_max
        self.channel_uses = channel_uses
        self.code_length = code_length
        self.info_length = info_length
        self.channel_type = channel_type

    @classmethod
    def from_training_arguments(cls, hf_args: HFTrainingArguments, **override_kwargs) -> "TrainingArguments":
        """Create TrainingArguments from standard TrainingArguments.

        Args:
            training_args: Standard TrainingArguments instance
            **override_kwargs: Additional arguments to override or add

        Returns:
            TrainingArguments instance
        """
        # Get all attributes from TrainingArguments
        args_dict = {}
        for key in dir(hf_args):
            if not key.startswith("_") and not callable(getattr(hf_args, key)):
                try:
                    args_dict[key] = getattr(hf_args, key)
                except (AttributeError, TypeError):
                    continue

        # Override with any additional kwargs
        args_dict.update(override_kwargs)

        # Filter valid parameters
        valid_params = cls._get_valid_parameters()
        filtered_args = {k: v for k, v in args_dict.items() if k in valid_params}

        return cls(**filtered_args)

    @classmethod
    def _get_valid_parameters(cls) -> set:
        """Get set of valid parameter names for this class."""
        # Get parameters from the class __init__ method
        import inspect

        init_signature = inspect.signature(cls.__init__)
        return set(init_signature.parameters.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()

        # Add communication-specific parameters
        comm_params = ["snr_min", "snr_max", "noise_variance_min", "noise_variance_max", "channel_uses", "code_length", "info_length", "channel_type"]

        for param in comm_params:
            if hasattr(self, param):
                result[param] = getattr(self, param)

        return result

    def to_hydra_config(self) -> Any:
        """Convert to Hydra DictConfig.

        Returns:
            DictConfig representation

        Raises:
            ImportError: If Hydra is not available
        """
        return OmegaConf.create(self.to_dict())

    def get_snr_range(self) -> tuple:
        """Get SNR range as tuple."""
        return (self.snr_min, self.snr_max)

    def get_noise_variance_range(self) -> tuple:
        """Get noise variance range as tuple."""
        return (self.noise_variance_min, self.noise_variance_max)

    def update_from_hydra(self, hydra_config: Union[Any, dict]) -> None:
        """Update current configuration from Hydra config.

        Args:
            hydra_config: Hydra DictConfig or dict to update from
        """
        if isinstance(hydra_config, DictConfig):
            config_dict = OmegaConf.to_container(hydra_config, resolve=True)
        else:
            config_dict = hydra_config

        # Extract training config if nested
        if "training" in config_dict:
            config_dict = config_dict["training"]

        # Update attributes
        valid_params = self._get_valid_parameters()
        for key, value in config_dict.items():
            if key in valid_params and hasattr(self, key):
                setattr(self, key, value)


class KairaTrainer(Trainer):
    """Unified trainer for all communication models.

    This trainer automatically adapts to different model types and supports multiple
    configuration systems for training arguments:
    - Hugging Face TrainingArguments
    - Hydra DictConfig
    - Plain Python dictionaries

    Models are responsible for their own configuration, channel simulation,
    constraints, and domain-specific logic via their config systems.

    The trainer focuses on training mechanics and automatically detects model
    types to apply appropriate loss functions. All domain-specific metrics
    should be handled by models or provided via compute_metrics parameter.
    """

    def __init__(self, model, args: Union[TrainingArguments, HFTrainingArguments, DictConfig, dict], loss_fn: Optional[str] = None, hydra_config: Optional[DictConfig] = None, **kwargs):
        """Initialize trainer.

        Args:
            model: BaseModel instance to train (handles domain-specific logic internally)
            args: Training arguments (TrainingArguments, HFTrainingArguments, DictConfig, or dict)
            loss_fn: Name of loss function from kaira.losses registry (auto-detected if None)
            hydra_config: Hydra configuration for training parameters only
            **kwargs: Additional arguments for base Trainer
        """
        # Convert args to custom TrainingArguments if needed
        if isinstance(args, TrainingArguments):
            training_args = args
        elif isinstance(args, HFTrainingArguments):
            training_args = TrainingArguments.from_training_arguments(args)
        elif isinstance(args, DictConfig):
            training_args = TrainingArguments.from_hydra(args)
        elif isinstance(args, dict):
            training_args = TrainingArguments.from_dict(args)
        else:
            training_args = self._convert_to_training_arguments(args)

        super().__init__(model=model, args=training_args, **kwargs)

        # Extract training-specific parameters from hydra config if provided
        training_config = {}
        if hydra_config is not None:
            training_config = _extract_config_value(hydra_config, "training", {})

        # Store training config for reference
        self.training_config = training_config
        self.hydra_config = hydra_config

        # Setup loss function
        self.loss_fn = self._setup_loss_function(loss_fn)

    def _setup_loss_function(self, loss_fn_name: Optional[str]):
        """Setup loss function from kaira.losses registry."""
        if loss_fn_name is not None:
            # Use specified loss function
            return LossRegistry.create(loss_fn_name)

        # Auto-detect based on model type
        model_name = self.model.__class__.__name__.lower()
        if "deepjscc" in model_name or "jscc" in model_name:
            return LossRegistry.create("mseloss")
        elif "fec" in model_name:
            return LossRegistry.create("crossentropyloss")
        else:
            # Default to MSE for reconstruction tasks
            return LossRegistry.create("mseloss")

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with automatic adaptation to model type."""
        # Extract labels/targets from inputs
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            # Try different possible label keys
            labels = inputs.get("info_bits") or inputs.get("source") or inputs.get("input_data")
            if labels is None:
                raise ValueError("No labels found in inputs. Expected 'labels', 'info_bits', 'source', or 'input_data'")

        # Forward pass through model
        outputs = model(**inputs)

        # Compute appropriate loss based on output type
        loss = self._compute_appropriate_loss(outputs, labels)

        return (loss, outputs) if return_outputs else loss

    def _compute_appropriate_loss(self, outputs, labels):
        """Compute loss using kaira.losses."""
        # Extract predictions from different output formats
        if hasattr(outputs, "logits"):
            predictions = outputs.logits
            # For classification tasks, flatten if needed
            if isinstance(self.loss_fn, LossRegistry.get("crossentropyloss")):
                predictions = predictions.view(-1, predictions.size(-1))
                labels = labels.view(-1).long()
        elif hasattr(outputs, "last_hidden_state"):
            predictions = outputs.last_hidden_state
        elif isinstance(outputs, torch.Tensor):
            predictions = outputs
        else:
            raise ValueError(f"Unknown output type: {type(outputs)}. Expected tensor or object with 'logits' or 'last_hidden_state'")

        return self.loss_fn(predictions, labels)

    def _convert_to_training_arguments(self, args: Union[DictConfig, dict]) -> TrainingArguments:
        """Convert various config types to TrainingArguments."""
        if isinstance(args, TrainingArguments):
            return args

        # Extract arguments as dict
        if isinstance(args, DictConfig):
            args_dict = OmegaConf.to_container(args, resolve=True)
        elif isinstance(args, dict):
            args_dict = args.copy()
        else:
            # Try to convert object to dict
            try:
                args_dict = vars(args)
            except TypeError:
                raise ValueError(f"Cannot convert args of type {type(args)} to TrainingArguments")

        # Filter out args that TrainingArguments doesn't accept
        training_args_keys = set(HFTrainingArguments.__init__.__code__.co_varnames)
        filtered_args = {k: v for k, v in args_dict.items() if k in training_args_keys}

        return TrainingArguments(**filtered_args)

    @classmethod
    def from_hydra_config(cls, hydra_cfg: DictConfig, model, **kwargs):
        """Create trainer from Hydra configuration."""
        # Extract training arguments from hydra config
        training_args = OmegaConf.select(hydra_cfg, "training", default={})

        # Extract loss function if specified
        loss_fn = OmegaConf.select(hydra_cfg, "loss.name", default=None)
        if loss_fn:
            kwargs["loss_fn"] = loss_fn

        return cls(model=model, args=training_args, hydra_config=hydra_cfg, **kwargs)

    @classmethod
    def from_training_args(cls, training_args: HFTrainingArguments, model, **kwargs):
        """Create trainer from Hugging Face TrainingArguments."""
        return cls(model=model, args=training_args, **kwargs)

    @classmethod
    def from_hydra_with_kaira_args(cls, hydra_cfg: DictConfig, model, **kwargs):
        """Create trainer from Hydra config using TrainingArguments."""
        # Create TrainingArguments from hydra config
        training_args = TrainingArguments.from_hydra(hydra_cfg)

        # Extract loss function if specified
        loss_fn = OmegaConf.select(hydra_cfg, "loss.name", default=None)
        if loss_fn:
            kwargs["loss_fn"] = loss_fn

        return cls(model=model, args=training_args, hydra_config=hydra_cfg, **kwargs)


__all__ = ["KairaTrainer", "TrainingArguments"]
