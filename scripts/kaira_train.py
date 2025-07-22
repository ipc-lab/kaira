#!/usr/bin/env python3
"""Kaira Training CLI.

Command-line interface for training Kaira communication models.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from kaira.models import BaseModel, ModelRegistry
from kaira.training import Trainer, TrainingArguments
from kaira.utils import seed_everything


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Kaira Training CLI - Train communication system models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  kaira-train --list-models

  # Train a specific model with default configuration
  kaira-train --model deepjscc --output-dir ./results

  # Train with custom configuration
  kaira-train --model deepjscc --output-dir ./results --epochs 20 --batch-size 64

  # Train with custom SNR range
  kaira-train --model channel_code --snr-min 0 --snr-max 15 --learning-rate 1e-3

  # Train with Hydra configuration file
  kaira-train --model deepjscc --config-file ./configs/training_example.yaml

  # Resume training from checkpoint
  kaira-train --model deepjscc --resume-from-checkpoint ./results/checkpoint-1000

  # Train and upload to Hugging Face Hub
  kaira-train --model deepjscc --push-to-hub --hub-model-id username/my-model

  # Train and upload to private Hub repository
  kaira-train --model deepjscc --push-to-hub --hub-model-id username/my-model --hub-private --hub-token your_token
        """,
    )

    # Main action arguments
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--list-models", action="store_true", help="List available models")
    action_group.add_argument("--model", type=str, help="Model to train")

    # Configuration
    parser.add_argument("--config-file", type=Path, help="Load training configuration from Hydra YAML file")

    # Training configuration
    parser.add_argument("--output-dir", type=Path, default="./training_results", help="Output directory for training results (default: ./training_results)")
    parser.add_argument("--epochs", "--num-train-epochs", type=float, dest="num_train_epochs", default=10.0, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", "--per-device-train-batch-size", type=int, dest="per_device_train_batch_size", default=32, help="Training batch size per device (default: 32)")
    parser.add_argument("--eval-batch-size", "--per-device-eval-batch-size", type=int, dest="per_device_eval_batch_size", default=32, help="Evaluation batch size per device (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Number of warmup steps (default: 1000)")

    # Communication-specific parameters
    parser.add_argument("--snr-min", type=float, default=0.0, help="Minimum SNR value for training (default: 0.0)")
    parser.add_argument("--snr-max", type=float, default=20.0, help="Maximum SNR value for training (default: 20.0)")
    parser.add_argument("--noise-variance-min", type=float, default=0.1, help="Minimum noise variance (default: 0.1)")
    parser.add_argument("--noise-variance-max", type=float, default=2.0, help="Maximum noise variance (default: 2.0)")
    parser.add_argument("--channel-uses", type=int, help="Number of channel uses")
    parser.add_argument("--code-length", type=int, help="Length of the code")
    parser.add_argument("--info-length", type=int, help="Length of information bits")
    parser.add_argument("--channel-type", type=str, default="awgn", help="Type of channel simulation (default: awgn)")

    # Training control
    parser.add_argument("--logging-steps", type=int, default=100, help="Log every X steps (default: 100)")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluate every X steps (default: 500)")
    parser.add_argument("--save-steps", type=int, default=1000, help="Save every X steps (default: 1000)")
    parser.add_argument("--eval-strategy", choices=["no", "steps", "epoch"], default="steps", help="Evaluation strategy (default: steps)")
    parser.add_argument("--save-strategy", choices=["no", "steps", "epoch"], default="steps", help="Save strategy (default: steps)")
    parser.add_argument("--save-total-limit", type=int, default=3, help="Maximum number of checkpoints to keep (default: 3)")

    # Data configuration
    parser.add_argument("--dataset", type=str, help="Dataset to use for training")
    parser.add_argument("--train-data-path", type=Path, help="Path to training data")
    parser.add_argument("--eval-data-path", type=Path, help="Path to evaluation data")
    parser.add_argument("--max-train-samples", type=int, help="Maximum number of training samples")
    parser.add_argument("--max-eval-samples", type=int, help="Maximum number of evaluation samples")

    # Checkpointing and resuming
    parser.add_argument("--resume-from-checkpoint", type=Path, help="Resume training from checkpoint")
    parser.add_argument("--overwrite-output-dir", action="store_true", help="Overwrite output directory if it exists")

    # Device and performance
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Computation device (default: auto)")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--dataloader-num-workers", type=int, default=0, help="Number of dataloader workers (default: 0)")

    # General options
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", action="store_true", help="Suppress output except errors")

    # Hugging Face Hub upload options
    hub_group = parser.add_argument_group("Hugging Face Hub Upload")
    hub_group.add_argument("--push-to-hub", action="store_true", help="Upload trained model to Hugging Face Hub")
    hub_group.add_argument("--hub-model-id", type=str, help="Model ID for Hugging Face Hub (e.g., 'username/model-name')")
    hub_group.add_argument("--hub-token", type=str, help="Hugging Face Hub authentication token (or set HF_TOKEN env var)")
    hub_group.add_argument("--hub-private", action="store_true", help="Make the Hub repository private")
    hub_group.add_argument("--hub-strategy", choices=["end", "checkpoint"], default="end", help="When to upload to Hub: 'end' (after training) or 'checkpoint' (during training) (default: end)")

    # Evaluation and testing
    parser.add_argument("--do-eval", action="store_true", help="Run evaluation during training")
    parser.add_argument("--do-predict", action="store_true", help="Run prediction after training")

    return parser


def list_available_models():
    """List available models."""
    print("Available Models:")
    models = ModelRegistry.list_models()
    if models:
        for model_name in sorted(models):
            model_class = ModelRegistry.get_model_cls(model_name)
            if model_class and hasattr(model_class, "__doc__") and model_class.__doc__:
                description = model_class.__doc__.split("\n")[0].strip()
            else:
                description = "Communication model"
            print(f"  - {model_name}: {description}")
    else:
        print("  No models available")
        print("  Make sure you have registered models in the ModelRegistry")


def load_model_from_config(model_name: str) -> BaseModel:
    """Load model from configuration."""
    # Get model class from registry
    model_class = ModelRegistry.get_model_cls(model_name)
    if model_class is None:
        available_models = ModelRegistry.list_models()
        raise ValueError(f"Unknown model '{model_name}'. Available models: {', '.join(available_models)}")

    # Create model instance with default configuration
    try:
        model = model_class()
    except Exception as e:
        print(f"Error creating model '{model_name}': {e}", file=sys.stderr)
        if hasattr(model_class, "__init__"):
            import inspect

            sig = inspect.signature(model_class.__init__)
            print(f"Model constructor signature: {sig}", file=sys.stderr)
        raise

    return model


def create_training_arguments_from_args(args) -> TrainingArguments:
    """Create training arguments from command-line arguments or config file."""
    if args.config_file:
        # Load Hydra configuration from file
        config = OmegaConf.load(args.config_file)
        training_args = TrainingArguments.from_hydra_config(config)
    else:
        # Create from CLI arguments using TrainingArguments method
        training_args = TrainingArguments.from_cli_args(args)

    return training_args


def load_datasets(args, training_args: TrainingArguments):
    """Load training and evaluation datasets."""
    train_dataset = None
    eval_dataset = None

    # For now, we rely on models to handle their own data generation
    # This is because communication models often generate synthetic data
    # based on their specific requirements (SNR ranges, modulation schemes, etc.)

    if args.dataset:
        print(f"Note: Dataset '{args.dataset}' specified, but models will handle data generation internally")

    if args.train_data_path:
        print(f"Note: Training data path '{args.train_data_path}' specified, but models will handle data generation internally")

    if args.eval_data_path:
        print(f"Note: Evaluation data path '{args.eval_data_path}' specified, but models will handle data generation internally")

    # Communication models typically generate data on-the-fly based on their configuration
    # The trainer will work with the model's internal data generation methods

    return train_dataset, eval_dataset


def setup_device(args):
    """Setup computation device."""
    import torch

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU", file=sys.stderr)
        device = "cpu"

    if not args.quiet:
        print(f"Using device: {device}")

    return device


def setup_hub_upload(args):
    """Setup Hugging Face Hub upload configuration."""
    if not args.push_to_hub:
        return None

    try:
        import os

        from huggingface_hub import login

        # Handle authentication
        token = args.hub_token or os.getenv("HF_TOKEN")
        if not token:
            print("Warning: No Hugging Face token provided. You may need to login manually.", file=sys.stderr)
            print("Set HF_TOKEN environment variable or use --hub-token argument", file=sys.stderr)
        else:
            try:
                login(token=token)
                if not args.quiet:
                    print("Successfully authenticated with Hugging Face Hub")
            except Exception as e:
                print(f"Warning: Failed to authenticate with Hugging Face Hub: {e}", file=sys.stderr)

        # Validate model ID
        if not args.hub_model_id:
            raise ValueError("--hub-model-id is required when using --push-to-hub")

        if "/" not in args.hub_model_id:
            raise ValueError("Hub model ID must be in format 'username/model-name'")

        return {
            "model_id": args.hub_model_id,
            "token": token,
            "private": args.hub_private,
            "strategy": args.hub_strategy,
        }

    except ImportError:
        print("Error: huggingface_hub is required for Hub upload. Install with: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error setting up Hub upload: {e}", file=sys.stderr)
        sys.exit(1)


def upload_to_hub(model, trainer, hub_config, args):
    """Upload model to Hugging Face Hub."""
    if not hub_config:
        return

    try:
        import tempfile

        import torch
        from huggingface_hub import HfApi

        if not args.quiet:
            print(f"Uploading model to Hugging Face Hub: {hub_config['model_id']}")

        api = HfApi(token=hub_config["token"])

        # Create repository if it doesn't exist
        try:
            api.create_repo(repo_id=hub_config["model_id"], exist_ok=True, private=hub_config["private"])
        except Exception as e:
            if not args.quiet:
                print(f"Repository may already exist: {e}")

        # Create a temporary directory for the model files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_model_dir = Path(temp_dir) / "model"
            temp_model_dir.mkdir()

            # Save model to temporary directory
            model_save_path = temp_model_dir / "pytorch_model.bin"
            torch.save(model.state_dict(), model_save_path)

            # Create model card
            model_card_content = f"""---
tags:
- kaira
- communication-systems
- deep-learning
library_name: kaira
license: mit
---

# {hub_config['model_id'].split('/')[-1]}

This model was trained using the Kaira framework for communication systems.

## Model Information

- Framework: Kaira
- Model Type: {args.model}
- Training Configuration: {args.output_dir}

## Usage

```python
import torch
from kaira.models import ModelRegistry

# Load the model
model_class = ModelRegistry.get_model_cls('{args.model}')
model = model_class()

# Load the trained weights
state_dict = torch.load('pytorch_model.bin')
model.load_state_dict(state_dict)
```

## Training Details

- Epochs: {getattr(args, 'num_train_epochs', 'N/A')}
- Batch Size: {getattr(args, 'per_device_train_batch_size', 'N/A')}
- Learning Rate: {getattr(args, 'learning_rate', 'N/A')}
- SNR Range: {getattr(args, 'snr_min', 'N/A')} to {getattr(args, 'snr_max', 'N/A')} dB

"""

            model_card_path = temp_model_dir / "README.md"
            with open(model_card_path, "w") as f:
                f.write(model_card_content)

            # Create config file with model information
            config_content = {
                "model_type": args.model,
                "framework": "kaira",
                "snr_min": getattr(args, "snr_min", None),
                "snr_max": getattr(args, "snr_max", None),
                "channel_type": getattr(args, "channel_type", None),
            }

            import json

            config_path = temp_model_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(config_content, f, indent=2)

            # Upload all files
            api.upload_folder(folder_path=str(temp_model_dir), repo_id=hub_config["model_id"], repo_type="model", commit_message=f"Upload {args.model} model trained with Kaira")

        if not args.quiet:
            print(f"✅ Successfully uploaded model to: https://huggingface.co/{hub_config['model_id']}")

    except Exception as e:
        print(f"Error uploading to Hub: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()


def train_model(model: BaseModel, training_args: TrainingArguments, train_dataset=None, eval_dataset=None, resume_from_checkpoint: Optional[Path] = None):
    """Train the model."""
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Start training
    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=str(resume_from_checkpoint))
    else:
        trainer.train()

    return trainer


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle list models command
    if args.list_models:
        list_available_models()
        return

    # Validate required arguments
    if not args.model:
        print("Error: --model is required when not listing models", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    # Set random seed
    seed_everything(args.seed)

    # Setup device
    setup_device(args)

    # Setup Hub upload if requested
    hub_config = setup_hub_upload(args)

    if not args.quiet:
        print(f"Training model: {args.model}")
        print(f"Output directory: {args.output_dir}")
        print(f"Random seed: {args.seed}")
        if hub_config:
            print(f"Will upload to Hub: {hub_config['model_id']}")

    try:
        # Load model
        if not args.quiet:
            print("Loading model...")
        model = load_model_from_config(args.model)

        # Create training arguments
        if not args.quiet:
            print("Setting up training configuration...")
        training_args = create_training_arguments_from_args(args)

        if args.verbose:
            print(f"Training arguments: {training_args.to_dict()}")

        # Load datasets
        if not args.quiet:
            print("Loading datasets...")
        train_dataset, eval_dataset = load_datasets(args, training_args)

        if train_dataset and not args.quiet:
            print(f"Training dataset size: {len(train_dataset)}")
        if eval_dataset and not args.quiet:
            print(f"Evaluation dataset size: {len(eval_dataset)}")

        # Note: Most communication models generate synthetic data internally
        # If no external dataset is provided, the model should handle data generation
        if not train_dataset and not args.quiet:
            print("Note: No external training dataset provided - model should handle data generation internally")

        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=args.overwrite_output_dir)

        # Train model
        if not args.quiet:
            print("Starting training...")
        trainer = train_model(model=model, training_args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, resume_from_checkpoint=args.resume_from_checkpoint)

        # Save final model
        if not args.quiet:
            print("Saving final model...")
        trainer.save_model()

        # Upload to Hub if requested
        if hub_config and hub_config["strategy"] == "end":
            upload_to_hub(model, trainer, hub_config, args)

        # Run final evaluation if requested
        if args.do_eval and eval_dataset:
            if not args.quiet:
                print("Running final evaluation...")
            eval_results = trainer.evaluate()
            print(f"Final evaluation results: {eval_results}")

        # Run prediction if requested
        if args.do_predict and eval_dataset:
            if not args.quiet:
                print("Running prediction...")
            predict_results = trainer.predict(eval_dataset)
            print(f"Prediction completed. Results shape: {predict_results.predictions.shape}")

        # Setup Hugging Face Hub upload if requested
        hub_config = setup_hub_upload(args)

        # Upload to Hugging Face Hub if configured
        if hub_config and hub_config["strategy"] == "end":
            upload_to_hub(model, trainer, hub_config, args)

        if not args.quiet:
            print("\nTraining completed successfully!")
            print(f"Model saved to: {args.output_dir}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
