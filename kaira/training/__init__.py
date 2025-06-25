"""Kaira training module.

This module provides training infrastructure for communication models, including:
- TrainingArguments: Flexible training arguments supporting multiple config systems
- Trainer: Unified trainer for all communication models

Examples:
    Basic usage with TrainingArguments:
    >>> from kaira.training import TrainingArguments, Trainer
    >>> args = TrainingArguments(output_dir="./results", num_train_epochs=10)
    >>> trainer = Trainer(model, args)

    Using Hydra configurations:
    >>> args = TrainingArguments.from_hydra(hydra_config)
    >>> trainer = Trainer.from_hydra_config(hydra_config, model)

    Direct dict configurations:
    >>> args = TrainingArguments.from_dict({"output_dir": "./results"})
    >>> trainer = Trainer(model, args)
"""

from .arguments import TrainingArguments
from .trainer import Trainer

__all__ = [
    "TrainingArguments",
    "Trainer",
]
