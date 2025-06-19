"""Training arguments for Kaira communication models.

This module provides training arguments that support Hydra configuration systems.
"""

from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments as HFTrainingArguments


class TrainingArguments(HFTrainingArguments):
    """Training arguments that support Hydra configuration management.

    This class extends transformers.TrainingArguments to provide seamless integration
    with Hydra configuration management while maintaining full compatibility with
    Hugging Face ecosystem. It supports:

    - Direct instantiation from Hydra DictConfig via from_hydra_config
    - Communication-specific parameters
    - Automatic parameter filtering and validation

    Examples:
        >>> # From Hydra config
        >>> hydra_config = OmegaConf.create({"training": {"output_dir": "./results", "num_train_epochs": 10}})
        >>> args = TrainingArguments.from_hydra_config(hydra_config)

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
    def from_hydra_config(cls, hydra_cfg: DictConfig, **override_kwargs) -> "TrainingArguments":
        """Create TrainingArguments from Hydra configuration.

        Args:
            hydra_cfg: Hydra DictConfig containing training configuration
            **override_kwargs: Additional arguments to override or add

        Returns:
            TrainingArguments instance
        """
        # Extract training-specific parameters from hydra config
        # If the config has a "training" key, use that, otherwise use the whole config
        if "training" in hydra_cfg:
            training_config = hydra_cfg.training
        else:
            training_config = hydra_cfg

        # Convert DictConfig to dict if needed
        if isinstance(training_config, DictConfig):
            training_config = OmegaConf.to_container(training_config, resolve=True)

        # Override with any additional kwargs
        training_config.update(override_kwargs)

        # Filter valid parameters
        valid_params = cls._get_valid_parameters()
        filtered_args = {k: v for k, v in training_config.items() if k in valid_params}

        return cls(**filtered_args)

    @classmethod
    def from_cli_args(cls, args) -> "TrainingArguments":
        """Create TrainingArguments from command-line arguments.

        Args:
            args: Parsed command-line arguments (from argparse)

        Returns:
            TrainingArguments instance
        """
        # Define parameter mappings with their expected types
        param_mappings = {
            # Standard training arguments
            "output_dir": str,
            "num_train_epochs": float,
            "per_device_train_batch_size": int,
            "per_device_eval_batch_size": int,
            "learning_rate": float,
            "warmup_steps": int,
            "logging_steps": int,
            "eval_steps": int,
            "save_steps": int,
            "eval_strategy": str,
            "save_strategy": str,
            "save_total_limit": int,
            "fp16": bool,
            "dataloader_num_workers": int,
            "do_eval": bool,
            "do_predict": bool,
            "overwrite_output_dir": bool,
            # Communication-specific parameters
            "snr_min": float,
            "snr_max": float,
            "noise_variance_min": float,
            "noise_variance_max": float,
            "channel_uses": int,
            "code_length": int,
            "info_length": int,
            "channel_type": str,
        }

        # Extract and convert arguments
        cli_args: Dict[str, Any] = {}
        for param_name, type_converter in param_mappings.items():
            if hasattr(args, param_name):
                value = getattr(args, param_name)
                if value is not None:
                    cli_args[param_name] = type_converter(value)

        return cls(**cli_args)

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
