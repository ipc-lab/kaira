"""Multiple Access Channel models for Kaira.

This module implements models for Multiple Access Channel (MAC) scenarios, where multiple devices
transmit data simultaneously over a shared wireless channel. It provides base classes and utilities
for implementing various MAC protocols and studying their performance.
"""

from typing import Any, List, Optional, Type, Union

import torch
from torch import nn

from kaira.channels import BaseChannel
from kaira.constraints import BaseConstraint
from kaira.models.base import BaseModel
from kaira.models.registry import ModelRegistry


@ModelRegistry.register_model("multiple_access_channel")
class MultipleAccessChannelModel(BaseModel):
    """A model simulating a Multiple Access Channel (MAC).

    In a MAC scenario, multiple transmitters (users) send signals simultaneously
    over a shared channel to a single receiver. The receiver then attempts to
    decode the messages from all users.

    This model supports both shared and separate encoders/decoders based on the
    provided configuration during initialization. A single decoder instance implies
    joint decoding.

    Attributes:
        encoders (nn.ModuleList): A list of encoder modules. Contains one shared encoder
            or one encoder per user.
        decoders (nn.ModuleList): A list of decoder modules. Contains one shared (joint)
            decoder or one decoder per user.
        channel (BaseChannel): The communication channel model.
        power_constraint (BaseConstraint): Power constraint applied to the sum of encoded signals.
        num_users (int): The number of users (transmitters).
    """

    def __init__(
        self,
        encoders: Union[Type[BaseModel], BaseModel, List[BaseModel], nn.ModuleList],
        decoders: Union[Type[BaseModel], BaseModel, List[BaseModel], nn.ModuleList],
        channel: BaseChannel,
        power_constraint: BaseConstraint,
        num_devices: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the MultipleAccessChannelModel.

        Args:
            encoders (Union[Type[BaseModel], BaseModel, List[BaseModel], nn.ModuleList]):
                Encoder configuration. Can be:
                - A class (Type[BaseModel]): An instance will be created for each device.
                - An instance (BaseModel): This instance will be shared across all devices.
                - A list/ModuleList of instances: One encoder per device. Length must match num_devices.
            decoders (Union[Type[BaseModel], BaseModel, List[BaseModel], nn.ModuleList]):
                Decoder configuration. Can be:
                - A class (Type[BaseModel]): A single instance will be created (joint decoder).
                - An instance (BaseModel): This instance will be used as the single joint decoder.
                - A list/ModuleList of instances: One decoder per device (separate decoding). Length must match num_devices.
            channel (BaseChannel): The channel model instance.
            power_constraint (BaseConstraint): The power constraint instance.
            num_devices (Optional[int]): The number of users/devices. Required if encoders/decoders
                are provided as single instances or classes. Inferred if encoders/decoders are lists.
            *args: Variable positional arguments passed to the base class and module instantiation.
            **kwargs: Variable keyword arguments passed to the base class and module instantiation.
        """
        super().__init__(*args, **kwargs)

        # --- Determine Number of Devices ---
        inferred_num_devices_enc = None
        if isinstance(encoders, (list, nn.ModuleList)):
            inferred_num_devices_enc = len(encoders)
            if num_devices is None:
                num_devices = inferred_num_devices_enc
            elif num_devices != inferred_num_devices_enc:
                raise ValueError(f"Provided num_devices ({num_devices}) does not match the number of encoders ({inferred_num_devices_enc}).")

        inferred_num_devices_dec = None
        if isinstance(decoders, (list, nn.ModuleList)):
            inferred_num_devices_dec = len(decoders)
            if num_devices is None:
                num_devices = inferred_num_devices_dec
            elif num_devices != inferred_num_devices_dec:
                # Allow single decoder in list for joint decoding
                # Check if it's the decoder case and only one decoder is provided
                is_single_joint_decoder_in_list = inferred_num_devices_dec == 1
                if not is_single_joint_decoder_in_list:
                    raise ValueError(f"Provided num_devices ({num_devices}) does not match the number of decoders ({inferred_num_devices_dec}).")

        # Check consistency if both were lists and neither was a single joint decoder
        if inferred_num_devices_enc is not None and inferred_num_devices_dec is not None and inferred_num_devices_dec != 1 and inferred_num_devices_enc != inferred_num_devices_dec:
            raise ValueError(f"Number of encoders ({inferred_num_devices_enc}) must match number of decoders ({inferred_num_devices_dec}) when both are provided as lists with more than one decoder.")

        if num_devices is None:
            # Try inferring from decoder if encoder wasn't a list but decoder was
            if inferred_num_devices_dec is not None:
                # If only one decoder was provided in the list, we still don't know num_devices
                if inferred_num_devices_dec != 1:
                    num_devices = inferred_num_devices_dec
                else:
                    # Need num_devices from encoder or explicit arg if decoder is single/joint
                    raise ValueError("num_devices must be specified if encoders are not provided as a list and only a single (joint) decoder is provided.")
            else:
                raise ValueError("num_devices must be specified if encoders and decoders are not provided as lists.")

        if not isinstance(num_devices, int) or num_devices <= 0:
            raise ValueError(f"num_devices must be a positive integer, got {num_devices}")

        self.num_users = num_devices
        self.num_devices = num_devices  # Keep for compatibility

        # --- Initialize Encoders ---
        # Pass *args, **kwargs to _initialize_modules
        self.encoders = self._initialize_modules(encoders, num_devices, "Encoder", *args, **kwargs)

        # --- Initialize Decoders ---
        # Pass *args, **kwargs to _initialize_modules
        self.decoders = self._initialize_modules(decoders, num_devices, "Decoder", *args, **kwargs)

        # --- Assign Channel and Constraint ---
        if not isinstance(channel, BaseChannel):
            raise TypeError(f"Channel must be an instance of BaseChannel, but got {type(channel)}")
        self.channel = channel

        if not isinstance(power_constraint, BaseConstraint):
            raise TypeError(f"Power constraint must be an instance of BaseConstraint, but got {type(power_constraint)}")
        self.power_constraint = power_constraint

    def _initialize_modules(self, module_config: Union[Type[BaseModel], BaseModel, List[BaseModel], nn.ModuleList], num_devices: int, module_name: str, *args: Any, **kwargs: Any) -> nn.ModuleList:
        """Helper function to initialize encoder or decoder modules."""
        modules_list = []
        is_shared = False  # Track if the module is intended to be shared

        if isinstance(module_config, (list, nn.ModuleList)):
            # Separate instances provided in a list
            if module_name == "Decoder" and len(module_config) == 1:
                # Special case: A list containing a single decoder implies joint decoding
                is_shared = True
                modules_list = list(module_config)
            elif len(module_config) != num_devices:
                raise ValueError(f"Number of {module_name.lower()}s in the list ({len(module_config)}) must match num_devices ({num_devices}).")
            else:
                # Correct number of separate instances provided
                modules_list = list(module_config)

        elif isinstance(module_config, nn.Module):
            # Single instance provided -> treat as shared
            is_shared = True
            instance = module_config
            if module_name == "Decoder":
                # For joint decoder, store only the single instance in the list
                modules_list = [instance]
            else:  # Encoders: replicate reference num_devices times for forward pass indexing
                modules_list = [instance] * num_devices

        elif isinstance(module_config, type):
            # Class provided
            module_cls = module_config
            if module_name == "Decoder":
                # Create one instance for joint decoding
                is_shared = True
                instance = module_cls(*args, **kwargs)
                modules_list = [instance]
            else:  # Encoders: Create separate instances
                is_shared = False  # Explicitly separate
                modules_list = [module_cls(*args, **kwargs) for _ in range(num_devices)]

        else:
            raise TypeError(f"Invalid type for {module_name.lower()} configuration: {type(module_config)}")

        # Validate all items are nn.Module
        for i, mod in enumerate(modules_list):
            if not isinstance(mod, nn.Module):
                raise TypeError(f"{module_name} at index {i} (or the shared instance) must be an instance of nn.Module, but got {type(mod)}")

        # Final check for encoder list length if shared instance was replicated
        if module_name == "Encoder" and is_shared and len(modules_list) != num_devices:
            # This should not happen with the current logic, but as a safeguard
            raise RuntimeError(f"Internal error: Shared encoder list length ({len(modules_list)}) doesn't match num_devices ({num_devices}).")
        # Final check for decoder list length
        if module_name == "Decoder" and not is_shared and len(modules_list) != num_devices:
            raise RuntimeError(f"Internal error: Separate decoder list length ({len(modules_list)}) doesn't match num_devices ({num_devices}).")
        if module_name == "Decoder" and is_shared and len(modules_list) != 1:
            raise RuntimeError(f"Internal error: Shared decoder list should have length 1, but got {len(modules_list)}.")

        # Return as ModuleList
        return nn.ModuleList(modules_list)

    def forward(self, x: List[torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the Multiple Access Channel model.

        Args:
            x (List[torch.Tensor]): A list of input tensors, one for each user.
                Each tensor should have shape (batch_size, message_dim).
            *args: Additional positional arguments passed to encoders, channel, and decoder(s).
            **kwargs: Additional keyword arguments passed to encoders, channel, and decoder(s).

        Returns:
            torch.Tensor: The output tensor from the decoder(s).
                If joint decoder: (batch_size, decoded_message_dim).
                If separate decoders: (batch_size, num_users * decoded_message_dim_per_user).
        """
        if not isinstance(x, list) or not all(isinstance(t, torch.Tensor) for t in x):
            # Added check for list input based on test_mac_model_invalid_function_call
            raise ValueError("Input 'x' must be a list of torch.Tensors.")

        if len(x) != self.num_users:
            raise ValueError(f"Number of input tensors ({len(x)}) must match the number of users ({self.num_users}).")

        if not self.encoders:
            raise ValueError("Encoders must be initialized before calling forward.")
        if not self.decoders:
            raise ValueError("Decoders must be initialized before calling forward.")

        # 1. Encode messages for each user
        encoded_signals = []
        # Determine if encoder is shared: list has 1 element, or list has num_users refs to the same object
        is_shared_encoder = len(self.encoders) == 1 or (self.num_users > 1 and len(self.encoders) == self.num_users and self.encoders[0] is self.encoders[1])

        for i in range(self.num_users):
            # Use index 0 if shared, otherwise use user index i
            # We need to ensure the index is valid for the actual list length
            actual_encoder_list_len = len(self.encoders)
            if is_shared_encoder and actual_encoder_list_len > 0:
                encoder_to_use_idx = 0  # Always use the first (and only unique) encoder if shared
            elif not is_shared_encoder and i < actual_encoder_list_len:
                encoder_to_use_idx = i
            else:
                # This case should ideally not happen due to init logic, but check defensively
                raise IndexError(f"Encoder index calculation error. is_shared={is_shared_encoder}, index={i}, list_len={actual_encoder_list_len}")

            encoder = self.encoders[encoder_to_use_idx]
            encoded_signals.append(encoder(x[i], *args, **kwargs))

        # 2. Combine encoded signals (summing them simulates superposition on the channel)
        combined_signal = torch.sum(torch.stack(encoded_signals), dim=0)

        # 3. Apply power constraint to the combined signal
        constrained_signal = self.power_constraint(combined_signal)

        # 4. Pass the combined signal through the channel
        # Pass *args and **kwargs to the channel
        received_signal = self.channel(constrained_signal, *args, **kwargs)

        # 5. Decode the received signal
        is_joint_decoder = len(self.decoders) == 1
        if is_joint_decoder:
            # Use the single joint decoder
            decoder = self.decoders[0]
            reconstructed_messages = decoder(received_signal, *args, **kwargs)
        else:
            # Use separate decoders
            # Ensure the number of decoders matches the number of users for separate decoding
            if len(self.decoders) != self.num_users:
                raise ValueError(f"Number of separate decoders ({len(self.decoders)}) must match num_users ({self.num_users}) for separate decoding mode.")

            reconstructed_signals_list = []  # Renamed to avoid confusion
            for i in range(self.num_users):
                decoder = self.decoders[i]
                # Assumption: Each separate decoder `i` processes the combined signal
                # to reconstruct the message for user `i`.
                reconstructed_signals_list.append(decoder(received_signal, *args, **kwargs))
            # Concatenate the outputs along the feature dimension
            reconstructed_messages = torch.cat(reconstructed_signals_list, dim=1)

        # Return the final reconstructed messages
        return reconstructed_messages
