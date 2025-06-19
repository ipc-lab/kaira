"""Training arguments for Kaira communication models.

This module provides flexible training arguments that support multiple configuration systems
including Hugging Face TrainingArguments, Hydra configurations, and plain dictionaries.
"""

from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments as HFTrainingArguments


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
        eval_strategy: str = "steps",
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
            eval_strategy: Evaluation strategy
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
            eval_strategy=eval_strategy,
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
            hf_args: Standard TrainingArguments instance
            **override_kwargs: Additional arguments to override or add

        Returns:
            TrainingArguments instance
        """
        # Get all attributes from TrainingArguments
        args_dict = {}
        for key in dir(hf_args):
            if not key.startswith("_"):
                try:
                    value = getattr(hf_args, key)
                    if not callable(value):
                        args_dict[key] = value
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

    @classmethod
    def from_hydra_config(cls, hydra_cfg: DictConfig, **override_kwargs) -> "TrainingArguments":
        """Create TrainingArguments from Hydra configuration.

        Args:
            hydra_cfg: Hydra DictConfig containing training configuration
            **override_kwargs: Additional arguments to override or add

        Returns:
            TrainingArguments instance
        """
        # Extract training-specific parameters from hydra config
        training_config = _extract_config_value(hydra_cfg, "training", {})

        # Override with any additional kwargs
        training_config.update(override_kwargs)

        return cls.from_dict(training_config)

    @classmethod
    def _convert_to_training_arguments(cls, args: Union[DictConfig, dict]) -> "TrainingArguments":
        """Convert various config types to TrainingArguments.

        Args:
            args: Configuration in various formats

        Returns:
            TrainingArguments instance
        """
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
        from transformers import TrainingArguments as HFTrainingArguments

        training_args_keys = set(HFTrainingArguments.__init__.__code__.co_varnames)
        filtered_args = {k: v for k, v in args_dict.items() if k in training_args_keys}

        return cls(**filtered_args)
