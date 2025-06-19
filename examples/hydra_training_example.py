#!/usr/bin/env python3
"""Example script showing how to use Hydra configurations with kaira-train."""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add the kaira root directory to the path so we can import modules
kaira_root = Path(__file__).parent.parent
sys.path.append(str(kaira_root))


@hydra.main(version_base=None, config_path="../configs", config_name="training_example")
def train_with_hydra(cfg: DictConfig) -> None:
    """Train a model using Hydra configuration."""
    print("=" * 50)
    print("Training with Hydra Configuration")
    print("=" * 50)

    # Print the loaded configuration
    print("Loaded configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Extract model name from config
    model_type = cfg.model.get("type", "deepjscc")

    try:
        # Load model from Hydra config
        print(f"\nLoading model: {model_type}")
        model_config = OmegaConf.to_container(cfg.model, resolve=True)

        # For demonstration, we'll use the ModelRegistry to create the model
        from kaira.models import ModelRegistry

        model_class = ModelRegistry.get_model_cls(model_type)

        # Remove type from config as it's not needed for model creation
        model_config.pop("type", None)
        model_config.pop("_target_", None)  # Remove Hydra target if present

        if model_config:
            model = model_class(**model_config)
        else:
            model = model_class()

        print(f"✓ Model loaded successfully: {model.__class__.__name__}")

        # Create training arguments from Hydra config
        print("\nSetting up training arguments...")
        training_args_dict = OmegaConf.to_container(cfg.training, resolve=True)

        from kaira.training import Trainer, TrainingArguments

        training_args = TrainingArguments.from_dict(training_args_dict)

        print("✓ Training arguments created")
        print(f"  - Output directory: {training_args.output_dir}")
        print(f"  - Epochs: {training_args.num_train_epochs}")
        print(f"  - Batch size: {training_args.per_device_train_batch_size}")
        print(f"  - Learning rate: {training_args.learning_rate}")
        print(f"  - SNR range: [{training_args.snr_min}, {training_args.snr_max}]")

        # Create trainer (for demonstration)
        print("\nCreating trainer...")
        _ = Trainer(
            model=model,
            args=training_args,
            train_dataset=None,  # Communication models typically generate data internally
            eval_dataset=None,
        )

        print("✓ Trainer created successfully")
        print("\nNote: This is a demonstration script.")
        print("To actually start training, uncomment the trainer.train() line below.")
        print("Make sure your model and configuration are properly set up for training.")

        # Uncomment the line below to actually start training
        # trainer.train()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nThis is expected if the model requires specific parameters or datasets.")
        print("This script demonstrates the configuration loading process.")


if __name__ == "__main__":
    train_with_hydra()
