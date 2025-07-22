"""Unified trainer for communication models using Transformers framework.

This module provides a flexible trainer that supports multiple configuration systems
for training arguments, while models handle their own configuration separately.

The trainer works with BaseModel instances and lets the models handle their own
channel simulation and constraints internally.

Examples:
    Using models with Hugging Face configurations:
    >>> from kaira import Trainer
    >>> from kaira.models import BaseModel, ModelConfig
    >>> from transformers import TrainingArguments
    >>>
    >>> # Configure the model
    >>> model_config = ModelConfig(input_dim=512, channel_uses=64)
    >>> model = BaseModel.from_pretrained_config(model_config)
    >>>
    >>> # Configure training
    >>> training_args = TrainingArguments(output_dir="./results", num_train_epochs=10)
    >>> trainer = Trainer(model, training_args)

    Using Hydra configurations:
    >>> # Model handles its own configuration
    >>> model = BaseModel.from_hydra_config(hydra_cfg.model)
    >>> trainer = Trainer.from_hydra_config(hydra_cfg, model)

    Using plain dictionaries:
    >>> # Model handles configuration internally
    >>> model = BaseModel.from_config({"input_dim": 512, "channel_uses": 64})
    >>> # Training config
    >>> training_config = {"output_dir": "./results", "num_train_epochs": 10}
    >>> trainer = Trainer(model, training_config)
"""

from typing import Optional, Union

from omegaconf import DictConfig
from transformers import Trainer as HFTrainer
from transformers import TrainingArguments as HFTrainingArguments

from .arguments import TrainingArguments


class Trainer(HFTrainer):
    """Unified trainer for all communication models.

    This trainer automatically adapts to different model types and supports multiple
    configuration systems for training arguments:
    - Hugging Face TrainingArguments
    - Kaira TrainingArguments
    - Hydra DictConfig
    - Plain Python dictionaries

    Models are responsible for their own configuration, channel simulation,
    constraints, and domain-specific logic via their config systems.

    The trainer focuses on training mechanics. All domain-specific metrics
    and loss functions should be provided by the user via the compute_metrics
    and loss function parameters.
    """

    def __init__(self, model, args: Union[TrainingArguments, HFTrainingArguments, DictConfig, dict], **kwargs):
        """Initialize trainer.

        Args:
            model: BaseModel instance to train (handles domain-specific logic internally)
            args: Training arguments (TrainingArguments, HFTrainingArguments, DictConfig, or dict)
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
            # Try to convert with the class method
            training_args = TrainingArguments._convert_to_training_arguments(args)

        super().__init__(model=model, args=training_args, **kwargs)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save model and optionally upload to Hub.

        Args:
            output_dir: Optional output directory
            _internal_call: Internal call flag
        """
        # Call parent save_model
        super().save_model(output_dir, _internal_call)

        # Check if we should upload to Hub on checkpoints
        if hasattr(self.args, "push_to_hub") and self.args.push_to_hub and hasattr(self.args, "hub_strategy") and self.args.hub_strategy == "checkpoint":
            try:
                self._upload_checkpoint_to_hub(output_dir)
            except Exception as e:
                print(f"Warning: Failed to upload checkpoint to Hub: {e}")

    def _upload_checkpoint_to_hub(self, output_dir: Optional[str] = None):
        """Upload checkpoint to Hugging Face Hub."""
        if not hasattr(self.args, "hub_model_id") or not self.args.hub_model_id:
            return

        try:
            from pathlib import Path

            from huggingface_hub import HfApi

            # Use provided output_dir or default to args.output_dir
            model_dir = Path(output_dir) if output_dir else Path(self.args.output_dir)

            api = HfApi(token=getattr(self.args, "hub_token", None))

            # Upload the checkpoint directory
            api.upload_folder(folder_path=str(model_dir), repo_id=self.args.hub_model_id, repo_type="model", commit_message=f"Upload checkpoint at step {self.state.global_step if hasattr(self, 'state') else 'unknown'}")

            print(f"✅ Uploaded checkpoint to Hub: {self.args.hub_model_id}")

        except ImportError:
            print("Warning: huggingface_hub not available for checkpoint upload")
        except Exception as e:
            print(f"Error uploading checkpoint to Hub: {e}")

    @classmethod
    def from_hydra_config(cls, hydra_cfg: DictConfig, model, **kwargs):
        """Create trainer from Hydra configuration."""
        # Create TrainingArguments from hydra config
        training_args = TrainingArguments.from_hydra_config(hydra_cfg)

        return cls(model=model, args=training_args, **kwargs)

    @classmethod
    def from_training_args(cls, training_args: HFTrainingArguments, model, **kwargs):
        """Create trainer from Hugging Face TrainingArguments."""
        return cls(model=model, args=training_args, **kwargs)
