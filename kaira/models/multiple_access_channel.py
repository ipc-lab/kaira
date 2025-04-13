"""Multiple Access Channel models for Kaira.

This module implements models for Multiple Access Channel (MAC) scenarios, where multiple devices
transmit data simultaneously over a shared wireless channel. It provides base classes and utilities
for implementing various MAC protocols and studying their performance.
"""

from typing import Any, List, Optional, Union

import torch
from torch import nn

from kaira.channels import BaseChannel
from kaira.constraints import BaseConstraint

from .base import BaseModel
from .registry import ModelRegistry


@ModelRegistry.register_model("multiple_access_channel")
class MultipleAccessChannelModel(BaseModel):
    """A model simulating a Multiple Access Channel (MAC).

    In a MAC scenario, multiple transmitters (users) send signals simultaneously
    over a shared channel to a single receiver. The receiver then attempts to
    decode the messages from all users.

    This model supports joint decoding, where a single decoder processes the
    combined received signal to estimate all users' messages.

    Attributes:
        encoders (nn.ModuleList): A list of encoder modules, one for each user.
        decoder (BaseModel): A single decoder module that processes the combined signal.
        channel (BaseChannel): The communication channel model.
        power_constraint (BaseConstraint): Power constraint applied to the sum of encoded signals.
        num_users (int): The number of users (transmitters).
    """

    def __init__(
        self,
        encoders: Union[nn.ModuleList, List[BaseModel]],
        decoder: BaseModel, # Expecting a single decoder instance for joint decoding
        channel: BaseChannel,
        power_constraint: BaseConstraint,
        num_devices: Optional[int] = None, # num_devices might be redundant if encoders list is given
        **kwargs: Any,
    ):
        """Initialize the MultipleAccessChannelModel.

        Args:
            encoders (Union[nn.ModuleList, List[BaseModel]]): A list containing the encoder
                module for each user.
            decoder (BaseModel): The single joint decoder module instance.
            channel (BaseChannel): The channel model instance.
            power_constraint (BaseConstraint): The power constraint instance.
            num_devices (Optional[int]): The number of users/devices. If None, it's inferred
                from the length of the encoders list.
            **kwargs: Additional keyword arguments (currently unused but kept for flexibility).
        """
        super().__init__()

        if not isinstance(encoders, nn.ModuleList):
            self.encoders = nn.ModuleList(encoders)
        else:
            self.encoders = encoders

        if num_devices is None:
            self.num_users = len(self.encoders)
        else:
            self.num_users = num_devices
            if self.num_users != len(self.encoders):
                raise ValueError(f"num_devices ({num_devices}) does not match the number of encoders ({len(self.encoders)}).")

        # Directly assign the provided decoder instance
        if not isinstance(decoder, nn.Module):
             raise TypeError(f"Decoder must be an instance of nn.Module or BaseModel, but got {type(decoder)}")
        self.decoder = decoder

        self.channel = channel
        self.power_constraint = power_constraint


    def forward(self, x: List[torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the Multiple Access Channel model.

        Args:
            x (List[torch.Tensor]): A list of input tensors, one for each user.
                Each tensor should have shape (batch_size, message_dim).
            *args: Additional positional arguments passed potentially to channel.
                   These are currently NOT passed to the decoder.
            **kwargs: Additional keyword arguments, commonly including 'snr' for the channel.
                      Only channel-specific kwargs are used.

        Returns:
            torch.Tensor: The output tensor from the joint decoder, typically containing
                the concatenated reconstructed messages for all users.
                Shape: (batch_size, num_users * decoded_message_dim_per_user).
        """
        if len(x) != self.num_users:
            raise ValueError(f"Number of input tensors ({len(x)}) must match the number of users ({self.num_users}).")

        # 1. Encode messages for each user
        encoded_signals = [encoder(user_input) for encoder, user_input in zip(self.encoders, x)]

        # 2. Combine encoded signals (summing them simulates superposition on the channel)
        combined_signal = torch.sum(torch.stack(encoded_signals), dim=0)

        # 3. Apply power constraint to the combined signal
        constrained_signal = self.power_constraint(combined_signal)

        # 4. Pass the combined signal through the channel
        # Pass relevant kwargs (like snr) to the channel
        # Note: *args are also passed here, assuming they are for the channel
        channel_kwargs = {k: v for k, v in kwargs.items() if k in self.channel.forward.__code__.co_varnames}
        received_signal = self.channel(constrained_signal, *args, **channel_kwargs)

        # 5. Decode the received signal using the single joint decoder
        # Only pass the received signal, as MLPDecoder.forward expects only 'x'.
        # If a different decoder type were used that accepts *args or **kwargs,
        # this line would need adjustment.
        reconstructed_messages = self.decoder(received_signal)

        # The joint decoder should output the concatenated messages
        return reconstructed_messages
