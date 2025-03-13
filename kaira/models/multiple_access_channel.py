"""Multiple Access Channel models for Kaira.

This module implements models for Multiple Access Channel (MAC) scenarios, where multiple devices
transmit data simultaneously over a shared wireless channel. It provides base classes and utilities
for implementing various MAC protocols and studying their performance.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn

from kaira.channels.base import BaseChannel
from kaira.constraints.base import BaseConstraint
from kaira.models.base import BaseModel
from kaira.models.registry import ModelRegistry


@ModelRegistry.register_model("mac")
class MultipleAccessChannelModel(BaseModel):
    """Base class for Multiple Access Channel (MAC) models.

    This class provides the foundation for implementing MAC protocols where multiple
    transmitters send data to a single receiver over a shared channel. It supports:
    - Flexible number of transmitting devices
    - Optional shared encoders/decoders across devices
    - Power constraints per device or total system
    - Various channel models for transmission
    - Configurable interference patterns between devices
    - Successive interference cancellation

    The model structure allows studying different MAC scenarios such as:
    - TDMA (Time Division Multiple Access)
    - FDMA (Frequency Division Multiple Access)
    - NOMA (Non-Orthogonal Multiple Access)
    - Random Access protocols

    Attributes:
        channel (BaseChannel): Channel model for signal transmission
        power_constraint (BaseConstraint): Power constraint applied to signals
        encoders (nn.ModuleList): List of encoder networks for each device
        decoders (nn.ModuleList): List of decoder networks for each device
        num_devices (int): Number of transmitting devices
        shared_encoder (bool): Whether devices share the same encoder
        shared_decoder (bool): Whether devices share the same decoder
    """

    def __init__(self, channel: BaseChannel, power_constraint: BaseConstraint, encoder: Optional[Type[BaseModel]] = None, decoder: Optional[Type[BaseModel]] = None, num_devices: int = 2, shared_encoder: bool = False, shared_decoder: bool = False, **kwargs: Any):
        """Initialize the MAC model.

        Args:
            channel: Channel model for transmission
            power_constraint: Power constraint applied to transmitted signals
            encoder: Optional encoder model class. If None, must be set later
            decoder: Optional decoder model class. If None, must be set later
            num_devices: Number of transmitting devices (default: 2)
            shared_encoder: Whether to use same encoder for all devices (default: False)
            shared_decoder: Whether to use same decoder for all devices (default: False)
            **kwargs: Additional arguments passed to encoder/decoder constructors
        """
        super().__init__()

        # Basic components
        self.channel = channel
        self.power_constraint = power_constraint
        self.num_devices = num_devices
        self.shared_encoder = shared_encoder
        self.shared_decoder = shared_decoder

        # Initialize encoders
        self.encoders = nn.ModuleList()
        if encoder is not None:
            if shared_encoder:
                shared_enc = encoder(**kwargs)
                self.encoders.extend([shared_enc for _ in range(num_devices)])
            else:
                self.encoders.extend([encoder(**kwargs) for _ in range(num_devices)])

        # Initialize decoders
        self.decoders = nn.ModuleList()
        if decoder is not None:
            if shared_decoder:
                shared_dec = decoder(**kwargs)
                self.decoders.extend([shared_dec for _ in range(num_devices)])
            else:
                self.decoders.extend([decoder(**kwargs) for _ in range(num_devices)])

    def encode(self, inputs: List[torch.Tensor], device_indices: Optional[List[int]] = None) -> List[torch.Tensor]:
        """Encode input signals for transmission.

        Applies the corresponding encoder to each device's input signal.

        Args:
            inputs: List of input tensors, one per device
            device_indices: Optional list of specific device indices to encode.
                If None, encodes all devices

        Returns:
            List of encoded signals ready for transmission

        Raises:
            ValueError: If number of inputs doesn't match number of devices
            IndexError: If any device index is invalid
        """
        if device_indices is None:
            device_indices = list(range(self.num_devices))

        # At this point device_indices is guaranteed to be a list, not None
        if len(inputs) != len(device_indices):
            raise ValueError(f"Number of inputs ({len(inputs)}) must match number of " f"device indices ({len(device_indices)})")

        encoded = []
        for i, x in zip(device_indices, inputs):
            if not 0 <= i < self.num_devices:
                raise IndexError(f"Invalid device index: {i}")
            encoded.append(self.encoders[i](x))
        return encoded

    def decode(self, received: torch.Tensor, device_indices: Optional[List[int]] = None) -> List[torch.Tensor]:
        """Decode received signals for each device.

        Args:
            received: Combined received signal from channel
            device_indices: Optional list of specific device indices to decode.
                If None, decodes all devices

        Returns:
            List of decoded signals, one per device

        Raises:
            IndexError: If any device index is invalid
        """
        if device_indices is None:
            device_indices = list(range(self.num_devices))

        decoded = []
        for i in device_indices:
            if not 0 <= i < self.num_devices:
                raise IndexError(f"Invalid device index: {i}")
            decoded.append(self.decoders[i](received))
        return decoded

    def forward(self, inputs: List[torch.Tensor], csi: Optional[torch.Tensor] = None, noise: Optional[torch.Tensor] = None, device_indices: Optional[List[int]] = None) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], Dict[str, Any]]]:
        """Process inputs through the complete MAC system.

        The forward pass consists of:
        1. Encoding inputs from each device
        2. Applying power constraints
        3. Transmitting through the channel (with optional CSI/noise)
        4. Decoding received signals

        Args:
            inputs: List of input tensors, one per device
            csi: Optional channel state information
            noise: Optional explicit noise tensor
            device_indices: Optional list of specific devices to process

        Returns:
            - List of decoded outputs if no intermediate values requested
            - Tuple of (outputs, intermediate_values) if return_intermediate=True

        Raises:
            ValueError: If inputs/indices don't match or components missing
        """
        # Input validation
        if len(inputs) == 0:
            raise ValueError("No inputs provided")
        if device_indices is not None and len(inputs) != len(device_indices):
            raise ValueError("Number of inputs must match number of device indices")
        if not self.encoders or not self.decoders:
            raise ValueError("Encoders and decoders must be initialized")

        # Encode
        encoded = self.encode(inputs, device_indices)

        # Apply power constraint
        constrained = [self.power_constraint(x) for x in encoded]

        # Combine signals and transmit through channel
        combined = sum(constrained)  # Simple superposition
        received = self.channel(combined, csi=csi, noise=noise)

        # Decode
        outputs = self.decode(received, device_indices)

        return outputs

    def set_encoder(self, encoder: Type[BaseModel], device_index: Optional[int] = None, **kwargs: Any) -> None:
        """Set or replace encoder for specific device(s).

        Args:
            encoder: Encoder model class to use
            device_index: Index of device to set encoder for. If None, sets for all
            **kwargs: Arguments passed to encoder constructor

        Raises:
            IndexError: If device_index is invalid
        """
        if device_index is not None:
            if not 0 <= device_index < self.num_devices:
                raise IndexError(f"Invalid device index: {device_index}")
            self.encoders[device_index] = encoder(**kwargs)
        else:
            if self.shared_encoder:
                shared_enc = encoder(**kwargs)
                self.encoders = nn.ModuleList([shared_enc for _ in range(self.num_devices)])
            else:
                self.encoders = nn.ModuleList([encoder(**kwargs) for _ in range(self.num_devices)])

    def set_decoder(self, decoder: Type[BaseModel], device_index: Optional[int] = None, **kwargs: Any) -> None:
        """Set or replace decoder for specific device(s).

        Args:
            decoder: Decoder model class to use
            device_index: Index of device to set decoder for. If None, sets for all
            **kwargs: Arguments passed to decoder constructor

        Raises:
            IndexError: If device_index is invalid
        """
        if device_index is not None:
            if not 0 <= device_index < self.num_devices:
                raise IndexError(f"Invalid device index: {device_index}")
            self.decoders[device_index] = decoder(**kwargs)
        else:
            if self.shared_decoder:
                shared_dec = decoder(**kwargs)
                self.decoders = nn.ModuleList([shared_dec for _ in range(self.num_devices)])
            else:
                self.decoders = nn.ModuleList([decoder(**kwargs) for _ in range(self.num_devices)])
