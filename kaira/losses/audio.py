"""Audio Losses module for Kaira.

This module contains various loss functions for training audio-based communication systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from .base import BaseLoss
from .registry import LossRegistry


@LossRegistry.register_loss()
class L1AudioLoss(BaseLoss):
    """L1 Audio Loss Module.

    This module calculates the L1 loss between the input and target audio signals.
    """

    def __init__(self):
        """Initialize the L1AudioLoss module."""
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass through the L1AudioLoss module.

        Args:
            x (torch.Tensor): The input audio tensor.
            target (torch.Tensor): The target audio tensor.

        Returns:
            torch.Tensor: The L1 loss between the input and target audio.
        """
        return self.l1(x, target)


@LossRegistry.register_loss()
class SpectralConvergenceLoss(BaseLoss):
    """Spectral Convergence Loss Module.

    This module calculates the spectral convergence loss between the input and target spectra.
    """

    def __init__(self):
        """Initialize the SpectralConvergenceLoss module."""
        super().__init__()

    def forward(self, x_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SpectralConvergenceLoss module.

        Args:
            x_mag (torch.Tensor): The magnitude of the input spectrum.
            target_mag (torch.Tensor): The magnitude of the target spectrum.

        Returns:
            torch.Tensor: The spectral convergence loss.
        """
        return torch.norm(target_mag - x_mag, p="fro") / torch.norm(target_mag, p="fro")


@LossRegistry.register_loss()
class LogSTFTMagnitudeLoss(BaseLoss):
    """Log STFT Magnitude Loss Module.

    This module calculates the log STFT magnitude loss between the input and target spectra.
    """

    def __init__(self):
        """Initialize the LogSTFTMagnitudeLoss module."""
        super().__init__()

    def forward(self, x_mag: torch.Tensor, target_mag: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LogSTFTMagnitudeLoss module.

        Args:
            x_mag (torch.Tensor): The magnitude of the input spectrum.
            target_mag (torch.Tensor): The magnitude of the target spectrum.

        Returns:
            torch.Tensor: The log STFT magnitude loss.
        """
        log_x_mag = torch.log(x_mag + 1e-7)
        log_target_mag = torch.log(target_mag + 1e-7)
        return F.l1_loss(log_x_mag, log_target_mag)


@LossRegistry.register_loss()
class STFTLoss(BaseLoss):
    """STFT Loss Module.

    This module calculates the STFT loss between the input and target audio signals, combining
    spectral convergence loss and log STFT magnitude loss.
    """

    def __init__(self, fft_size=1024, hop_size=256, win_length=1024, window="hann"):
        """Initialize the STFTLoss module.

        Args:
            fft_size (int): FFT size for STFT. Default is 1024.
            hop_size (int): Hop size for STFT. Default is 256.
            win_length (int): Window length for STFT. Default is 1024.
            window (str): Window function type. Default is 'hann'.
        """
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = window
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass through the STFTLoss module.

        Args:
            x (torch.Tensor): The input audio tensor.
            target (torch.Tensor): The target audio tensor.

        Returns:
            torch.Tensor: The combined STFT loss.
        """
        window_fn = getattr(torch, f"{self.window}_window")
        window = window_fn(self.win_length, dtype=x.dtype, device=x.device)

        x_stft = torch.stft(
            x,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )

        target_stft = torch.stft(
            target,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )

        x_mag = torch.abs(x_stft)
        target_mag = torch.abs(target_stft)

        sc_loss = self.spectral_convergence_loss(x_mag, target_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, target_mag)

        return sc_loss + mag_loss


@LossRegistry.register_loss()
class MultiResolutionSTFTLoss(BaseLoss):
    """Multi-Resolution STFT Loss Module.

    This module calculates STFT loss at multiple resolutions for better time-frequency coverage.
    """

    def __init__(
        self,
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[128, 256, 512],
        win_lengths=[512, 1024, 2048],
        window="hann",
    ):
        """Initialize the MultiResolutionSTFTLoss module.

        Args:
            fft_sizes (list): List of FFT sizes for each resolution. Default is [512, 1024, 2048].
            hop_sizes (list): List of hop sizes for each resolution. Default is [128, 256, 512].
            win_lengths (list): List of window lengths for each resolution. Default is [512, 1024, 2048].
            window (str): Window function type. Default is 'hann'.
        """
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

        self.stft_losses = nn.ModuleList([STFTLoss(fft_size=fft_size, hop_size=hop_size, win_length=win_length, window=window) for fft_size, hop_size, win_length in zip(fft_sizes, hop_sizes, win_lengths)])

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MultiResolutionSTFTLoss module.

        Args:
            x (torch.Tensor): The input audio tensor.
            target (torch.Tensor): The target audio tensor.

        Returns:
            torch.Tensor: The multi-resolution STFT loss.
        """
        loss = 0.0
        for stft_loss in self.stft_losses:
            loss += stft_loss(x, target)

        return loss / len(self.stft_losses)


@LossRegistry.register_loss()
class MelSpectrogramLoss(BaseLoss):
    """Mel-Spectrogram Loss Module.

    This module calculates the loss between mel-spectrograms of input and target audio.
    """

    def __init__(
        self,
        sample_rate=22050,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        f_min=0.0,
        f_max=8000.0,
        log_mel=True,
    ):
        """Initialize the MelSpectrogramLoss module.

        Args:
            sample_rate (int): Audio sample rate. Default is 22050.
            n_fft (int): FFT size. Default is 1024.
            hop_length (int): Hop size. Default is 256.
            n_mels (int): Number of mel bands. Default is 80.
            f_min (float): Minimum frequency. Default is 0.0.
            f_max (float): Maximum frequency. Default is 8000.0.
            log_mel (bool): Whether to use log-mel spectrogram. Default is True.
        """
        super().__init__()
        self.melspec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
        self.log_mel = log_mel

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MelSpectrogramLoss module.

        Args:
            x (torch.Tensor): The input audio tensor.
            target (torch.Tensor): The target audio tensor.

        Returns:
            torch.Tensor: The mel-spectrogram loss.
        """
        x_mel = self.melspec_transform(x)
        target_mel = self.melspec_transform(target)

        if self.log_mel:
            x_mel = torch.log(x_mel + 1e-7)
            target_mel = torch.log(target_mel + 1e-7)

        return F.l1_loss(x_mel, target_mel)


@LossRegistry.register_loss()
class FeatureMatchingLoss(BaseLoss):
    """Feature Matching Loss Module.

    This module calculates the loss between features extracted from a pretrained model.
    """

    def __init__(self, model, layers, weights=None):
        """Initialize the FeatureMatchingLoss module.

        Args:
            model (BaseLoss): Pretrained model for feature extraction.
            layers (list): List of layer indices to extract features from.
            weights (list, optional): Weights for each layer. Default is None (equal weights).
        """
        super().__init__()
        self.model = model
        self.model.eval()
        self.layers = layers

        if weights is None:
            self.weights = [1.0] * len(layers)
        else:
            assert len(weights) == len(layers)
            self.weights = weights

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass through the FeatureMatchingLoss module.

        Args:
            x (torch.Tensor): The input audio tensor.
            target (torch.Tensor): The target audio tensor.

        Returns:
            torch.Tensor: The feature matching loss.
        """
        # Create tensors that require gradient
        x_with_grad = x.detach().requires_grad_(True)
        target_with_grad = target.detach().requires_grad_(True)

        # Register hooks to capture activations
        activations_x = {}
        activations_target = {}

        def get_activation(name):
            def hook(model, input, output):
                # Don't detach to allow gradient flow
                activations_x[name] = output

            return hook

        def get_target_activation(name):
            def hook(model, input, output):
                # Don't detach to allow gradient flow
                activations_target[name] = output

            return hook

        # Register hooks
        handles = []
        for i, layer_idx in enumerate(self.layers):
            handles.append(list(self.model.children())[layer_idx].register_forward_hook(get_activation(f"layer_{i}")))

        # Forward pass for input
        self.model(x_with_grad)

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Register hooks for target
        handles = []
        for i, layer_idx in enumerate(self.layers):
            handles.append(list(self.model.children())[layer_idx].register_forward_hook(get_target_activation(f"layer_{i}")))

        # Forward pass for target
        self.model(target_with_grad)

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Calculate loss
        loss = 0.0
        for i in range(len(self.layers)):
            layer_name = f"layer_{i}"
            # Use features from activations
            # We only detach the target activations to prevent training signal
            # from affecting the feature extractor
            loss += self.weights[i] * F.l1_loss(activations_x[layer_name], activations_target[layer_name].detach())

        return loss


@LossRegistry.register_loss()
class AudioContrastiveLoss(BaseLoss):
    """Audio Contrastive Loss Module.

    This module calculates a contrastive loss to bring similar audio samples closer in feature
    space. It can be used for self-supervised learning of audio representations.
    """

    def __init__(self, margin=1.0, temperature=0.1, normalize=True, reduction="mean"):
        """Initialize the AudioContrastiveLoss module.

        Args:
            margin (float): Margin for contrastive loss. Default is 1.0.
            temperature (float): Temperature scaling factor. Default is 0.1.
            normalize (bool): Whether to normalize features. Default is True.
            reduction (str): Reduction method ('mean', 'sum', 'none'). Default is 'mean'.
        """
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.normalize = normalize
        self.reduction = reduction

    def forward(self, features: torch.Tensor, target: torch.Tensor = None, projector=None, view_maker=None, labels=None) -> torch.Tensor:
        """Forward pass through the AudioContrastiveLoss module.

        Args:
            features (torch.Tensor): Audio feature embeddings.
            target (torch.Tensor, optional): Target features for comparison. If None, features
                are compared with themselves (self-supervised). Default is None.
            projector (nn.Module, optional): Optional projection network to map features
                to a lower-dimensional space. Default is None.
            view_maker (callable, optional): Function to create different views of the same
                data. Default is None.
            labels (torch.Tensor, optional): Labels for supervised contrastive learning.
                Default is None.

        Returns:
            torch.Tensor: The contrastive loss.
        """
        # Apply projector if provided
        if projector is not None:
            features = projector(features)
            if target is not None:
                target = projector(target)

        # Apply view maker if provided
        if view_maker is not None:
            # Create positive pairs using the view maker
            if target is None:
                target = view_maker(features)
            else:
                target = view_maker(target)

        # If no target is provided, use the features themselves
        if target is None:
            target = features

        # Normalize features
        if self.normalize:
            features = F.normalize(features, p=2, dim=1)
            target = F.normalize(target, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, target.t()) / self.temperature

        # Create mask for positive pairs
        if labels is not None:
            # Supervised contrastive learning with provided labels
            mask_positive = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()
        else:
            # Self-supervised learning (positive pairs are along the diagonal)
            batch_size = features.size(0)
            mask_positive = torch.eye(batch_size, device=features.device)

        # Remove self-comparisons for robustness
        mask_self = torch.eye(mask_positive.shape[0], device=mask_positive.device)
        mask_positive = mask_positive - mask_self

        # Compute loss (InfoNCE / NT-Xent loss)
        exp_logits = torch.exp(similarity_matrix) * (1 - mask_self)
        log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-10)

        # Compute mean of positive pairs
        # Handle the case where there are no positive pairs for some samples
        positive_per_sample = mask_positive.sum(1)
        # Avoid division by zero (add small epsilon)
        positive_per_sample = torch.clamp(positive_per_sample, min=1e-10)
        mean_log_prob_pos = (mask_positive * log_prob).sum(1) / positive_per_sample

        # Apply reduction
        if self.reduction == "mean":
            loss = -mean_log_prob_pos.mean()
        elif self.reduction == "sum":
            loss = -mean_log_prob_pos.sum()
        else:
            loss = -mean_log_prob_pos

        return loss


__all__ = ["L1AudioLoss", "SpectralConvergenceLoss", "LogSTFTMagnitudeLoss", "STFTLoss", "MultiResolutionSTFTLoss", "MelSpectrogramLoss", "FeatureMatchingLoss", "AudioContrastiveLoss"]
