"""DeepJSCC with Feedback implementation based on Kurka et al. 2020.

This module implements the Deep Joint Source-Channel Coding (DeepJSCC)
with feedback architecture proposed in :cite:p:`kurka2020deepjscc`.
The implementation supports both base layer transmission and refinement
layers for iterative image quality improvement.
"""


from typing import Any

import torch
import torch.nn as nn
from compressai.layers import GDN

from kaira.models.feedback_channel import FeedbackChannelModel
from kaira.models.registry import ModelRegistry


@ModelRegistry.register_model()
class DeepJSCCFeedbackEncoder(nn.Module):
    """Encoder network for DeepJSCC with Feedback :cite:`kurka2020deepjscc`.

    This encoder compresses the input image into a latent representation
    that can be transmitted through a noisy channel. The architecture uses
    a series of convolutional layers with GDN activations to efficiently
    encode visual information.

    Args:
        conv_depth (int): Depth of the output convolutional features, which
            determines the channel bandwidth usage.
    """

    def __init__(self, conv_depth):
        super().__init__()
        num_filters = 256

        # Sequential layer implementation
        self.layers = nn.ModuleList(
            [
                # Layer 0
                nn.Conv2d(3, num_filters, kernel_size=9, stride=2, padding=4, bias=True),
                GDN(num_filters),
                nn.PReLU(num_parameters=1),
                # Layer 1
                nn.Conv2d(num_filters, num_filters, kernel_size=5, stride=2, padding=2, bias=True),
                GDN(num_filters),
                nn.PReLU(num_parameters=1),
                # Layer 2
                nn.Conv2d(num_filters, num_filters, kernel_size=5, stride=1, padding=2, bias=True),
                GDN(num_filters),
                nn.PReLU(num_parameters=1),
                # Layer 3
                nn.Conv2d(num_filters, num_filters, kernel_size=5, stride=1, padding=2, bias=True),
                GDN(num_filters),
                nn.PReLU(num_parameters=1),
                # Output Layer
                nn.Conv2d(num_filters, conv_depth, kernel_size=5, stride=1, padding=2, bias=True),
            ]
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input image tensor of shape [B, C, H, W].
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            torch.Tensor: Encoded representation ready for channel transmission.
        """
        for layer in self.layers:
            x = layer(x)
        return x


@ModelRegistry.register_model()
class DeepJSCCFeedbackDecoder(nn.Module):
    """Decoder network for DeepJSCC with Feedback :cite:`kurka2020deepjscc`.

    This decoder reconstructs the image from the received noisy channel output.
    The architecture uses transposed convolutions with inverse GDN activations
    to convert the channel signal back into an image.

    Args:
        n_channels (int): Number of channels in the output image (typically 3 for RGB).
    """

    def __init__(self, n_channels):
        super().__init__()
        num_filters = 256

        # Sequential layer implementation
        self.layers = nn.ModuleList(
            [
                # Layer out
                nn.ConvTranspose2d(num_filters, num_filters, kernel_size=5, stride=1, padding=2, bias=True),
                GDN(num_filters, inverse=True),
                nn.PReLU(num_parameters=1),
                # Layer 0
                nn.ConvTranspose2d(num_filters, num_filters, kernel_size=5, stride=1, padding=2, bias=True),
                GDN(num_filters, inverse=True),
                nn.PReLU(num_parameters=1),
                # Layer 1
                nn.ConvTranspose2d(num_filters, num_filters, kernel_size=5, stride=1, padding=2, bias=True),
                GDN(num_filters, inverse=True),
                nn.PReLU(num_parameters=1),
                # Layer 2
                nn.ConvTranspose2d(num_filters, num_filters, kernel_size=5, stride=2, padding=2, output_padding=1, bias=True),
                GDN(num_filters, inverse=True),
                nn.PReLU(num_parameters=1),
                # Layer 3
                nn.ConvTranspose2d(num_filters, n_channels, kernel_size=9, stride=2, padding=4, output_padding=1, bias=True),
                nn.Sigmoid(),
            ]
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the decoder.

        Args:
            x (torch.Tensor): Channel output tensor to be decoded.
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            torch.Tensor: Reconstructed image in range [0, 1].
        """
        for layer in self.layers:
            x = layer(x)
        return x


class OutputsCombiner(nn.Module):
    """Combines previous outputs with residuals for iterative refinement :cite:`kurka2020deepjscc`.

    This module is used both for feedback generation and for processing feedback to improve image
    reconstruction quality. It takes a previous reconstruction and a residual signal, then produces
    an enhanced reconstruction through a small neural network.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 48, kernel_size=3, stride=1, padding=1)
        self.prelu1 = nn.PReLU(num_parameters=1)
        self.conv2 = nn.Conv2d(48, 3, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Combines previous reconstruction with residual information.

        Args:
            inputs (tuple): Contains:
                - img_prev (torch.Tensor): Previous reconstruction image
                - residual (torch.Tensor): Residual information for refinement
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            torch.Tensor: Enhanced reconstruction after combining inputs.
        """
        img_prev, residual = inputs

        # Concatenate previous image and residual
        reconst = torch.cat([img_prev, residual], dim=1)

        # Apply convolutions
        reconst = self.conv1(reconst)
        reconst = self.prelu1(reconst)
        reconst = self.conv2(reconst)
        reconst = self.sigmoid(reconst)

        return reconst


@ModelRegistry.register_model("deepjscc_feedback")
class DeepJSCCFeedbackModel(FeedbackChannelModel):
    """Deep Joint Source-Channel Coding with Feedback implementation :cite:`kurka2020deepjscc`.

    This model implements the DeepJSCC with feedback architecture from Kurka et al. 2020,
    which uses channel feedback to enhance image transmission quality in wireless channels.
    The model supports multiple iterations of feedback to progressively refine the
    reconstruction quality at the receiver.

    Args:
        channel_snr (float): Signal-to-noise ratio of the forward channel in dB.
        conv_depth (int): Depth of the convolutional features, controls bandwidth usage.
        channel_type (str): Type of channel ('awgn', 'fading', etc.).
        feedback_snr (float): Signal-to-noise ratio of the feedback channel in dB.
            If None, assumes perfect feedback.
        refinement_layer (bool): Whether this is a refinement layer (True) or
            base layer (False).
        layer_id (int): ID of the current layer for multi-layer configurations.
        forward_channel (BaseChannel, optional): The forward channel model. Defaults to None.
        feedback_channel (BaseChannel, optional): The feedback channel model. Defaults to None.
        target_analysis (bool, optional): Whether to perform target analysis. Defaults to False.
        max_iterations (int, optional): Maximum number of feedback iterations. Defaults to 3.
    """

    def __init__(
        self,
        channel_snr,
        conv_depth,
        channel_type,
        feedback_snr,
        refinement_layer,
        layer_id,
        forward_channel=None,
        feedback_channel=None,
        target_analysis=False,
        max_iterations=3,
    ):
        n_channels = 3  # change this if working with BW images
        self.refinement_layer = refinement_layer
        self.feedback_snr = feedback_snr
        self.layer = layer_id
        self.conv_depth = conv_depth

        # Define encoder and decoder
        encoder = DeepJSCCFeedbackEncoder(conv_depth)
        decoder = DeepJSCCFeedbackDecoder(n_channels)

        # Create feedback components
        feedback_generator = OutputsCombiner()
        feedback_processor = OutputsCombiner()

        # Initialize the pipeline
        super().__init__(
            encoder=encoder,
            forward_channel=forward_channel,
            decoder=decoder,
            feedback_generator=feedback_generator,
            feedback_channel=feedback_channel,
            feedback_processor=feedback_processor,
            max_iterations=max_iterations,
        )

        # Store parameters
        self.target_analysis = target_analysis

    def forward(self, inputs, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Forward pass of the DeepJSCC Feedback model.

        Processes the input through the encoder, channel, and decoder,
        potentially with multiple rounds of feedback. Handles both base layer
        and refinement layer cases.

        Args:
            inputs: Either:
                - For base layer: the input image tensor of shape [B, C, H, W]
                - For refinement layer: a tuple containing (input_image, previous_feedback_image,
                  previous_feedback_channel_output, previous_decoded_image,
                  previous_decoded_channel_output, previous_channel_gain)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            dict[str, Any]: Dictionary containing:
                - 'decoded_img': Reconstructed image
                - 'decoded_img_fb': Reconstructed image using feedback
                - 'channel_output': Channel output used for decoding
                - 'feedback_channel_output': Feedback channel output
                - 'channel_gain': Channel gain if applicable
        """
        if self.refinement_layer:
            (
                img,
                prev_img_out_fb,
                prev_chn_out_fb,
                prev_img_out_dec,
                prev_chn_out_dec,
                prev_chn_gain,
            ) = inputs
            # Concatenate previous feedback image with original image
            img_in = torch.cat([prev_img_out_fb, img], dim=1)
        else:  # base layer
            # inputs is just the original image
            img_in = img = inputs
            prev_chn_gain = None

        # Encode the input
        chn_in = self.encoder(img_in)

        # Pass through the channel
        chn_out, avg_power, chn_gain = self.forward_channel((chn_in, prev_chn_gain))

        # Add feedback noise to channel output
        if self.feedback_snr is None:  # No feedback noise
            chn_out_fb = chn_out
        else:
            # Use feedback channel for noisy feedback
            chn_out_fb, _, _ = self.feedback_channel((chn_out, None))

        if self.refinement_layer:
            # Combine channel output with previous stored channel outputs
            chn_out_exp = torch.cat([chn_out, prev_chn_out_dec], dim=1)
            residual_img = self.decoder(chn_out_exp)
            # Combine residual with previous stored image reconstruction
            decoded_img = self.feedback_processor((prev_img_out_dec, residual_img))

            # Feedback estimation
            chn_out_exp_fb = torch.cat([chn_out_fb, prev_chn_out_fb], dim=1)
            residual_img_fb = self.decoder(chn_out_exp_fb)
            decoded_img_fb = self.feedback_processor((prev_img_out_fb, residual_img_fb))
        else:
            chn_out_exp = chn_out
            decoded_img = self.decoder(chn_out_exp)
            chn_out_exp_fb = chn_out_fb
            decoded_img_fb = self.decoder(chn_out_exp_fb)

        return {"decoded_img": decoded_img, "decoded_img_fb": decoded_img_fb, "channel_output": chn_out_exp, "feedback_channel_output": chn_out_exp_fb, "channel_gain": chn_gain}
