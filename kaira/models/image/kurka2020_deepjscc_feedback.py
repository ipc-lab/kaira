"""
DeepJSCC with Feedback implementation based on Kurka et al. 2020.

This module implements the Deep Joint Source-Channel Coding (DeepJSCC)
with feedback architecture proposed in :cite:p:`kurka2020deepjscc`.
"""

import os
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from compressai.layers import GDN
from kaira.models.base import BaseModel, FeedbackChannelModel
from kaira.models.registry import ModelRegistry
from kaira.channels.base import BaseChannel


@ModelRegistry.register_model()
class DeepJSCCFeedbackEncoder(nn.Module):
    """Encoder network for DeepJSCC with Feedback :cite:`kurka2020deepjscc`.
    
    This encoder compresses the input image into a latent representation
    that can be transmitted through a noisy channel.
    
    Args:
        conv_depth (int): Depth of the output convolutional features.
    """
    def __init__(self, conv_depth):
        super(DeepJSCCFeedbackEncoder, self).__init__()
        num_filters = 256
        
        # Sequential layer implementation
        self.layers = nn.ModuleList([
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
            nn.Conv2d(num_filters, conv_depth, kernel_size=5, stride=1, padding=2, bias=True)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@ModelRegistry.register_model()
class DeepJSCCFeedbackDecoder(nn.Module):
    """Decoder network for DeepJSCC with Feedback :cite:`kurka2020deepjscc`.
    
    This decoder reconstructs the image from the received noisy channel output.
    
    Args:
        n_channels (int): Number of channels in the output image (typically 3 for RGB).
    """
    def __init__(self, n_channels):
        super(DeepJSCCFeedbackDecoder, self).__init__()
        num_filters = 256
        
        # Sequential layer implementation
        self.layers = nn.ModuleList([
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
            nn.Sigmoid()
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class OutputsCombiner(nn.Module):
    """Combines previous outputs with residuals for iterative refinement :cite:`kurka2020deepjscc`.
    
    This module is used both for feedback generation and for processing 
    feedback to improve image reconstruction quality.
    """
    def __init__(self):
        super(OutputsCombiner, self).__init__()
        self.conv1 = nn.Conv2d(6, 48, kernel_size=3, stride=1, padding=1)
        self.prelu1 = nn.PReLU(num_parameters=1)
        self.conv2 = nn.Conv2d(48, 3, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
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
    
    Args:
        channel_snr (float): Signal-to-noise ratio of the forward channel in dB.
        conv_depth (int): Depth of the convolutional features.
        channel_type (str): Type of channel ('awgn', 'fading', etc.).
        feedback_snr (float): Signal-to-noise ratio of the feedback channel in dB.
        refinement_layer (bool): Whether this is a refinement layer.
        layer_id (int): ID of the current layer.
        forward_channel (BaseChannel): The forward channel model.
        feedback_channel (BaseChannel): The feedback channel model.
        target_analysis (bool): Whether to perform target analysis.
        max_iterations (int): Maximum number of feedback iterations.
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
        super(DeepJSCCFeedbackModel, self).__init__(
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

    def forward(self, inputs):
        """Forward pass of the DeepJSCC Feedback model.
        
        Processes the input through the encoder, channel, and decoder,
        potentially with multiple rounds of feedback.
        
        Args:
            inputs: Either the input image (for base layer) or a tuple containing
                   the input image and previous feedback information.
        
        Returns:
            tuple: Contains the decoded image, feedback image, channel outputs,
                  feedback channel outputs, and channel gain.
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
        
        return (decoded_img, decoded_img_fb, chn_out_exp, chn_out_exp_fb, chn_gain)
