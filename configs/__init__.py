"""Configuration classes for Kaira models.

This module provides configuration classes that integrate with Hugging Face's PretrainedConfig
system while supporting communication-specific parameters.
"""

from typing import Optional

from transformers import PretrainedConfig, TrainingArguments


class KairaBaseConfig(PretrainedConfig):
    """Base configuration class for all Kaira training configurations.

    This class provides common configuration parameters that can be inherited by specific model
    configurations. It establishes a consistent interface for all configuration classes in the
    Kaira framework.

    Inherits from transformers.PretrainedConfig for compatibility with Hugging Face ecosystem and
    provides serialization/deserialization capabilities.
    """

    model_type = "kaira_base"

    def __init__(self, learning_rate: float = 1e-4, batch_size: int = 32, num_epochs: int = 100, warmup_steps: int = 1000, channel_type: str = "awgn", hidden_dim: int = 256, **kwargs):
        """Initialize KairaBaseConfig.

        Args:
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps
            channel_type: Type of channel simulation
            hidden_dim: Hidden dimension size
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)

        # Training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps

        # Channel simulation
        self.channel_type = channel_type

        # Model parameters
        self.hidden_dim = hidden_dim


class CommunicationTrainingArguments(TrainingArguments):
    """Training arguments for communication models."""

    def __init__(self, snr_min: float = 0.0, snr_max: float = 20.0, noise_variance_min: float = 0.1, noise_variance_max: float = 2.0, channel_uses: Optional[int] = None, code_length: Optional[int] = None, info_length: Optional[int] = None, **kwargs):
        """Initialize CommunicationTrainingArguments.

        Args:
            snr_min: Minimum SNR value for training
            snr_max: Maximum SNR value for training
            noise_variance_min: Minimum noise variance
            noise_variance_max: Maximum noise variance
            channel_uses: Number of channel uses
            code_length: Length of the code
            info_length: Length of information bits
            **kwargs: Additional arguments passed to TrainingArguments
        """
        super().__init__(**kwargs)

        # Channel simulation parameters
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.noise_variance_min = noise_variance_min
        self.noise_variance_max = noise_variance_max

        # Communication-specific parameters
        self.channel_uses = channel_uses
        self.code_length = code_length
        self.info_length = info_length


class FECConfig(KairaBaseConfig):
    """Configuration for Forward Error Correction models."""

    model_type = "fec"

    def __init__(self, info_length: int = 64, code_length: int = 128, code_rate: float = 0.5, num_encoder_layers: int = 4, num_decoder_layers: int = 4, num_attention_heads: int = 8, noise_variance_range: tuple = (0.1, 2.0), **kwargs):
        """Initialize FECConfig.

        Args:
            info_length: Length of information bits
            code_length: Length of encoded bits
            code_rate: Code rate (info_length / code_length)
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            num_attention_heads: Number of attention heads
            noise_variance_range: Range of noise variance for training
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)

        # Code parameters
        self.info_length = info_length
        self.code_length = code_length
        self.code_rate = code_rate

        # Model parameters
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_attention_heads = num_attention_heads

        # Channel simulation
        self.noise_variance_range = noise_variance_range


def create_fec_training_args(config: FECConfig, output_dir: str = "./results") -> CommunicationTrainingArguments:
    """Create training arguments for FEC."""
    return CommunicationTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=100,
        save_steps=1000,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="ber",
        greater_is_better=False,
        noise_variance_min=config.noise_variance_range[0],
        noise_variance_max=config.noise_variance_range[1],
        code_length=config.code_length,
        info_length=config.info_length,
    )


__all__ = ["KairaBaseConfig", "CommunicationTrainingArguments", "FECConfig", "create_fec_training_args"]
