"""Implementation of the Discrete Task-Oriented Deep JSCC model.

This module implements the Discrete Task-Oriented Deep JSCC (DT-DeepJSCC) model
as proposed in :cite:`xie2023robust`. The model uses a discrete bottleneck for
robust task-oriented semantic communications, particularly for image classification
tasks under varying channel conditions.

Adapted from: https://github.com/SongjieXie/Discrete-TaskOriented-JSCC
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kaira.models.registry import ModelRegistry

from ..base import BaseModel


class Resblock(nn.Module):
    """Residual block for feature extraction in DT-DeepJSCC.

    This implements a standard residual block with two convolutional layers
    and a skip connection, used in the encoder network.

    Args:
        in_channels (int): Number of input channels
    """

    def __init__(self, in_channels):
        super().__init__()
        self.model = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(True), nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(in_channels), nn.ReLU(True), nn.Conv2d(in_channels, in_channels, 1, bias=False))

    def forward(self, x):
        """Forward pass for residual block.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width]

        Returns:
            torch.Tensor: Output tensor of the same shape as input
        """
        return x + self.model(x)


class Resblock_down(nn.Module):
    """Residual block with downsampling for feature extraction in DT-DeepJSCC.

    This implements a residual block that reduces spatial dimensions while
    potentially increasing channel dimensions.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(True), nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True), nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, 2, bias=False))

    def forward(self, x):
        """Forward pass for downsampling residual block.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width]

        Returns:
            torch.Tensor: Output tensor with downsampled spatial dimensions
                [batch_size, out_channels, height/2, width/2]
        """
        return self.downsample(x) + self.model(x)


class MaskAttentionSampler(nn.Module):
    """Mask attention sampler for discrete bottleneck in DT-DeepJSCC.

    This class implements the discrete bottleneck mechanism that maps continuous
    features to a discrete latent space using a learnable embedding table.
    During training, it uses Gumbel-Softmax trick for differentiable sampling.

    Args:
        dim_dic (int): Dimension of the feature vectors
        num_embeddings (int, optional): Number of embeddings in the codebook. Defaults to 50.

    References:
        :cite:`xie2023robust`
    """

    def __init__(self, dim_dic, num_embeddings=50):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dim_dic = dim_dic

        self.embedding = nn.Parameter(torch.Tensor(num_embeddings, dim_dic))
        nn.init.uniform_(self.embedding, -1 / num_embeddings, 1 / num_embeddings)

    def compute_score(self, X):
        """Compute attention scores between input features and embeddings.

        Args:
            X (torch.Tensor): Input feature tensor [batch_size*h*w, dim_dic]

        Returns:
            torch.Tensor: Attention scores [batch_size*h*w, num_embeddings]
        """
        return torch.matmul(X, self.embedding.transpose(1, 0)) / np.sqrt(self.dim_dic)

    def sample(self, score):
        """Sample from the discrete codebook based on attention scores.

        During training, uses Gumbel-Softmax for differentiable sampling.
        During inference, uses argmax for hard selection.

        Args:
            score (torch.Tensor): Attention scores [batch_size*h*w, num_embeddings]

        Returns:
            tuple:
                - torch.Tensor: Symbol indices [batch_size*h*w]
                - torch.Tensor: Softmax distribution over codebook [batch_size*h*w, num_embeddings]
        """
        dist = F.softmax(score, dim=-1)

        # During training, use Gumbel-Softmax for differentiable sampling
        if self.training:
            samples = F.gumbel_softmax(score, tau=0.5, hard=True)
            indices = torch.argmax(samples, dim=-1)
        else:
            # During inference, use hard selection
            indices = torch.argmax(score, dim=-1)

        return indices, dist

    def recover_features(self, indices):
        """Recover features from discrete indices using the embedding table.

        Args:
            indices (torch.Tensor): Symbol indices [batch_size*h*w]

        Returns:
            torch.Tensor: Recovered feature vectors [batch_size*h*w, dim_dic]
        """
        one_hot = F.one_hot(indices, num_classes=self.num_embeddings).float()
        out = torch.matmul(one_hot, self.embedding)
        return out

    def forward(self, X):
        """Forward pass for the mask attention sampler.

        Args:
            X (torch.Tensor): Input feature tensor [batch_size*h*w, dim_dic]

        Returns:
            tuple:
                - torch.Tensor: Symbol indices [batch_size*h*w]
                - torch.Tensor: Distribution over codebook [batch_size*h*w, num_embeddings]
        """
        score = self.compute_score(X)
        indices, dist = self.sample(score)
        return indices, dist


@ModelRegistry.register_model()
class Xie2023DTDeepJSCCEncoder(BaseModel):
    """Discrete Task-Oriented Deep JSCC encoder.

    This implements the encoder part of the DT-DeepJSCC architecture as described
    in :cite:`xie2023robust`. It maps input images to discrete latent representations
    that are robust to channel impairments.

    Args:
        architecture (str, optional): Type of architecture to use. Defaults to 'cifar10'.
                                     Options: 'cifar10' or 'custom'.
        in_channels (int): Number of input image channels (3 for RGB, 1 for grayscale)
        latent_channels (int): Number of channels in the latent representation
        num_embeddings (int, optional): Size of the discrete codebook. Defaults to None
                                       (automatically determined by architecture).
        input_size (tuple, optional): Input image size as (height, width). Defaults to None
                                     (automatically determined by architecture).

    Returns:
        Encoded discrete representation of the input.

    References:
        :cite:`xie2023robust`
    """

    def __init__(self, in_channels, latent_channels, architecture="cifar10", num_embeddings=None, input_size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.architecture = architecture.lower()
        self.latent_d = latent_channels

        # Set defaults based on architecture
        if self.architecture == "cifar10":
            self.input_size = (32, 32) if input_size is None else input_size
            self.num_embeddings = 400 if num_embeddings is None else num_embeddings
            self._build_cifar10_encoder(in_channels, latent_channels)
        elif self.architecture == "custom":
            if input_size is None:
                raise ValueError("Input size must be provided for custom architecture")
            self.input_size = input_size
            self.num_embeddings = 50 if num_embeddings is None else num_embeddings
            self._build_custom_encoder(in_channels, latent_channels)
        else:
            raise ValueError(f"Unknown architecture: {architecture}. " f"Choose from 'cifar10' or 'custom'")

    def _build_cifar10_encoder(self, in_channels, latent_channels):
        """Build CNN encoder suitable for CIFAR-10 sized images.

        Args:
            in_channels (int): Number of input channels
            latent_channels (int): Number of latent channels
        """
        self.prep = nn.Sequential(nn.Conv2d(in_channels, latent_channels // 8, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(latent_channels // 8), nn.ReLU())
        self.layer1 = nn.Sequential(nn.Conv2d(latent_channels // 8, latent_channels // 4, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(latent_channels // 4), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
        self.layer2 = nn.Sequential(nn.Conv2d(latent_channels // 4, latent_channels // 2, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(latent_channels // 2), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(nn.Conv2d(latent_channels // 2, latent_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(latent_channels), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))

        self.encoder = nn.Sequential(
            self.prep, self.layer1, Resblock(latent_channels // 4), self.layer2, self.layer3, Resblock(latent_channels)  # latent_channels//8 x 32 x 32  # latent_channels//4 x 16 x 16  # latent_channels//4 x 16 x 16  # latent_channels//2 x 8 x 8  # latent_channels x 4 x 4  # latent_channels x 4 x 4
        )
        self.sampler = MaskAttentionSampler(latent_channels, self.num_embeddings)
        self.is_convolutional = True

    def _build_custom_encoder(self, in_channels, latent_channels):
        """Build a custom encoder based on input_size.

        This is similar to the CIFAR-10 encoder but adapts to custom input sizes.

        Args:
            in_channels (int): Number of input channels
            latent_channels (int): Number of latent channels
        """
        # Only support convolutional architecture
        self._build_cifar10_encoder(in_channels, latent_channels)

    def forward(self, x):
        """Forward pass for the DT-DeepJSCC encoder.

        Args:
            x (torch.Tensor): Input image tensor [batch_size, channels, height, width]

        Returns:
            torch.Tensor: Bits representation [batch_size, h*w, bits_per_symbol]
        """
        features = self.encoder(x)

        # Reshape to apply the discrete bottleneck
        b, c, h, w = features.shape
        features = features.permute(0, 2, 3, 1).contiguous()
        features = features.view(-1, self.latent_d)

        # Apply the discrete bottleneck to get indices
        indices, _ = self.sampler(features)

        # Convert indices to bits
        bits_per_symbol = int(np.log2(self.num_embeddings))
        bits = torch.zeros((indices.size(0), bits_per_symbol), device=indices.device, dtype=torch.float)

        # Convert indices to bits representation
        for i in range(bits_per_symbol):
            bits[:, i] = ((indices >> i) & 1).float()

        # Store the spatial dimensions in the bits tensor itself
        # by reshaping to [batch_size, h*w, bits_per_symbol]
        bits = bits.view(b, h * w, bits_per_symbol)

        return bits


@ModelRegistry.register_model()
class Xie2023DTDeepJSCCDecoder(BaseModel):
    """Discrete Task-Oriented Deep JSCC decoder.

    This implements the decoder part of the DT-DeepJSCC architecture as described
    in :cite:`xie2023robust`. It maps discrete latent representations back to
    class predictions.

    Args:
        architecture (str, optional): Type of architecture to use. Defaults to 'cifar10'.
                                     Options: 'cifar10' or 'custom'.
        latent_channels (int): Number of channels in the latent representation
        out_classes (int): Number of output classes
        num_embeddings (int, optional): Size of the discrete codebook. Defaults to None
                                       (automatically determined by architecture).

    References:
        :cite:`xie2023robust`
    """

    def __init__(self, latent_channels, out_classes, architecture="cifar10", num_embeddings=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.architecture = architecture.lower()
        self.latent_d = latent_channels
        self.out_classes = out_classes

        # Set defaults based on architecture
        if self.architecture == "cifar10":
            self.num_embeddings = 400 if num_embeddings is None else num_embeddings
            self._build_cifar10_decoder(latent_channels, out_classes)
        elif self.architecture == "custom":
            self.num_embeddings = 50 if num_embeddings is None else num_embeddings
            self._build_custom_decoder(latent_channels, out_classes)
        else:
            raise ValueError(f"Unknown architecture: {architecture}. " f"Choose from 'cifar10' or 'custom'")

        # Create the sampler for feature recovery
        self.sampler = MaskAttentionSampler(latent_channels, self.num_embeddings)

    def _build_cifar10_decoder(self, latent_channels, out_classes):
        """Build CNN decoder suitable for CIFAR-10 architecture.

        Args:
            latent_channels (int): Number of latent channels
            out_classes (int): Number of output classes
        """
        self.decoder = nn.Sequential(Resblock(latent_channels), Resblock(latent_channels), nn.BatchNorm2d(latent_channels), nn.ReLU(), nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(latent_channels, out_classes))

    def _build_custom_decoder(self, latent_channels, out_classes):
        """Build custom decoder based on architecture type.

        Args:
            latent_channels (int): Number of latent channels
            out_classes (int): Number of output classes
        """
        # Default to CIFAR-10 architecture
        self._build_cifar10_decoder(latent_channels, out_classes)

    def forward(self, received_bits):
        """Forward pass for the DT-DeepJSCC decoder.

        Args:
            received_bits (torch.Tensor): Received bits from the channel [batch_size, h*w, bits_per_symbol]

        Returns:
            torch.Tensor: Class logits [batch_size, out_classes]
        """
        # Extract batch size from the received bits tensor
        batch_size = received_bits.size(0)
        bits_per_symbol = int(np.log2(self.num_embeddings))

        # Flatten the spatial dimension for processing
        received_bits_flat = received_bits.view(-1, bits_per_symbol)

        # Convert bits back to indices
        indices = torch.zeros(received_bits_flat.size(0), device=received_bits_flat.device, dtype=torch.long)
        for i in range(bits_per_symbol):
            indices = indices | ((received_bits_flat[:, i] > 0.5).long() << i)

        # Recover features from discrete symbols
        features = self.sampler.recover_features(indices)

        # The total number of spatial elements for each batch
        spatial_elements = received_bits.size(1)

        # Calculate appropriate spatial dimensions
        # For CIFAR-10, typically 4x4
        spatial_dim = int(np.sqrt(spatial_elements))

        # Handle non-square spatial dimensions if needed
        if spatial_dim * spatial_dim != spatial_elements:
            # Find factors for non-square feature maps
            for i in range(int(np.sqrt(spatial_elements)), 0, -1):
                if spatial_elements % i == 0:
                    h_dim = i
                    w_dim = spatial_elements // i
                    # Reshape with non-square dimensions
                    features = features.view(batch_size, h_dim, w_dim, self.latent_d)
                    features = features.permute(0, 3, 1, 2).contiguous()
                    return self.decoder(features)

        # Reshape to proper dimensions with batch size preserved
        features = features.view(batch_size, spatial_dim, spatial_dim, self.latent_d)
        features = features.permute(0, 3, 1, 2).contiguous()

        # Generate class logits
        return self.decoder(features)
