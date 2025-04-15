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

    This model supports joint decoding by default, where a single decoder processes the
    combined received signal to estimate all users' messages. It can also be
    configured with individual decoders per user or a shared decoder instance.

    Attributes:
        encoders (nn.ModuleList): A list of encoder modules, one for each user, or a single shared encoder.
        decoders (nn.ModuleList): A list of decoder modules, one for each user, or a single shared decoder.
        channel (BaseChannel): The communication channel model.
        power_constraint (BaseConstraint): Power constraint applied to the sum of encoded signals.
        num_users (int): The number of users (transmitters).
        shared_encoder (bool): Flag indicating if a single encoder instance is shared across users.
        shared_decoder (bool): Flag indicating if a single decoder instance is shared across users.
    """

    def __init__(
        self,
        encoders: Union[Type[BaseModel], BaseModel, List[BaseModel], nn.ModuleList],
        decoder: Union[Type[BaseModel], BaseModel, List[BaseModel], nn.ModuleList],  # Accept list, type, or instance
        channel: BaseChannel,
        power_constraint: BaseConstraint,
        num_devices: Optional[int] = None,
        shared_encoder: bool = False,
        shared_decoder: bool = False,  # Added shared_decoder flag
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the MultipleAccessChannelModel.

        Args:
            encoders (Union[Type[BaseModel], BaseModel, List[BaseModel], nn.ModuleList]):
                Encoder configuration. Can be:
                - A class (Type[BaseModel]): An instance will be created for each device unless shared_encoder=True.
                - An instance (BaseModel): This instance will be shared if shared_encoder=True, otherwise error.
                - A list/ModuleList of instances: One encoder per device. shared_encoder must be False.
            decoder (Union[Type[BaseModel], BaseModel, List[BaseModel], nn.ModuleList]):
                Decoder configuration. Can be:
                - A class (Type[BaseModel]): An instance will be created for each device unless shared_decoder=True.
                - An instance (BaseModel): This instance will be shared if shared_decoder=True, otherwise error.
                - A list/ModuleList of instances: One decoder per device. shared_decoder must be False.
            channel (BaseChannel): The channel model instance.
            power_constraint (BaseConstraint): The power constraint instance.
            num_devices (Optional[int]): The number of users/devices. If None, it's inferred
                from the length of the encoders list (if provided as a list). Required if
                encoder/decoder are provided as classes or single instances.
            shared_encoder (bool): If True, use a single shared encoder instance for all devices.
                Requires `encoders` to be a class or a single instance. Defaults to False.
            shared_decoder (bool): If True, use a single shared decoder instance for all devices.
                Requires `decoder` to be a class or a single instance. Defaults to False.
            *args: Variable positional arguments passed to the base class and potentially module instantiation.
            **kwargs: Variable keyword arguments passed to the base class and potentially module instantiation.
        """
        super().__init__(*args, **kwargs)  # Pass *args, **kwargs to base BaseModel

        # --- Determine Number of Devices ---
        if isinstance(encoders, (list, nn.ModuleList)):
            inferred_num_devices = len(encoders)
            if num_devices is None:
                num_devices = inferred_num_devices
            elif num_devices != inferred_num_devices:
                raise ValueError(f"Provided num_devices ({num_devices}) does not match the number of encoders ({inferred_num_devices}).")
            if shared_encoder:
                raise ValueError("shared_encoder cannot be True when encoders is provided as a list.")
        elif isinstance(decoder, (list, nn.ModuleList)):  # Check decoder list length
            inferred_num_devices = len(decoder)
            if num_devices is None:
                num_devices = inferred_num_devices
            elif num_devices != inferred_num_devices:
                raise ValueError(f"Provided num_devices ({num_devices}) does not match the number of decoders ({inferred_num_devices}).")
            if shared_decoder:
                raise ValueError("shared_decoder cannot be True when decoder is provided as a list.")

        if num_devices is None:
            raise ValueError("num_devices must be specified if encoders and decoder are not provided as lists.")

        self.num_users = num_devices
        self.num_devices = num_devices  # Keep for compatibility

        # --- Initialize Encoders ---
        self.shared_encoder = shared_encoder
        # Pass *args, **kwargs to _initialize_modules
        self.encoders = self._initialize_modules(encoders, num_devices, shared_encoder, "Encoder", *args, **kwargs)

        # --- Initialize Decoders ---
        self.shared_decoder = shared_decoder
        # Pass *args, **kwargs to _initialize_modules
        self.decoders = self._initialize_modules(decoder, num_devices, shared_decoder, "Decoder", *args, **kwargs)

        # --- Assign Channel and Constraint ---
        if not isinstance(channel, BaseChannel):
            raise TypeError(f"Channel must be an instance of BaseChannel, but got {type(channel)}")
        self.channel = channel

        if not isinstance(power_constraint, BaseConstraint):
            raise TypeError(f"Power constraint must be an instance of BaseConstraint, but got {type(power_constraint)}")
        self.power_constraint = power_constraint

    def _initialize_modules(self, module_config: Union[Type[BaseModel], BaseModel, List[BaseModel], nn.ModuleList], num_devices: int, shared: bool, module_name: str, *args: Any, **kwargs: Any) -> nn.ModuleList:
        """Helper function to initialize encoder or decoder modules."""
        modules_list = []
        if isinstance(module_config, (list, nn.ModuleList)):
            if shared:
                raise ValueError(f"shared_{module_name.lower()} cannot be True when {module_name.lower()}s is provided as a list.")
            if len(module_config) != num_devices:
                raise ValueError(f"Number of {module_name.lower()}s in the list ({len(module_config)}) must match num_devices ({num_devices}).")
            modules_list = list(module_config)  # Ensure it's a standard list before ModuleList
        elif isinstance(module_config, nn.Module):  # Single instance provided
            if shared:
                # Replicate the single instance for the list
                instance = module_config
                # If shared, we expect only one logical decoder, but store it in a list for consistency.
                # The forward pass will handle using the correct one (self.decoders[0]).
                # For multiple devices with a shared component, we store the same instance multiple times
                # only if the component itself needs to be accessed individually later (which is not the case here for the base forward).
                # Let's store just one instance if shared=True for simplicity, as only decoders[0] is used by default.
                # Update: Storing multiple references to the same object is fine and clearer.
                modules_list = [instance] * num_devices
            else:
                # This case is ambiguous - did the user intend to share or provide only one for a single-device setup?
                # If num_devices > 1, raise error. If num_devices == 1, allow it.
                if num_devices == 1:
                    modules_list = [module_config]
                else:
                    raise ValueError(f"A single {module_name} instance was provided, but shared_{module_name.lower()}=False and num_devices={num_devices}. " f"Set shared_{module_name.lower()}=True if you intend to share the instance, " f"or provide a class/list if you need separate instances.")
        elif isinstance(module_config, type):  # Class provided
            module_cls = module_config
            if shared:
                # Create one instance and replicate it
                instance = module_cls(*args, **kwargs)  # Pass *args, **kwargs here
                modules_list = [instance] * num_devices
            else:
                # Create a new instance for each device
                modules_list = [module_cls(*args, **kwargs) for _ in range(num_devices)]  # Pass *args, **kwargs here
        else:
            raise TypeError(f"Invalid type for {module_name.lower()} configuration: {type(module_config)}")

        # Validate all items are nn.Module
        for i, mod in enumerate(modules_list):
            if not isinstance(mod, nn.Module):
                # Check if it's the replicated shared instance; only need to check the first one
                if shared and i > 0 and mod is modules_list[0]:
                    continue
                raise TypeError(f"{module_name} at index {i} (or the shared instance) must be an instance of nn.Module, but got {type(mod)}")

        # If shared=True, the list contains multiple references to the *same* object.
        # If shared=False, the list contains references to *different* objects (either from input list or newly created).
        return nn.ModuleList(modules_list)

    def forward(self, x: List[torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the Multiple Access Channel model.

        Args:
            x (List[torch.Tensor]): A list of input tensors, one for each user.
                Each tensor should have shape (batch_size, message_dim).
            *args: Additional positional arguments passed to encoders, channel, and decoder.
            **kwargs: Additional keyword arguments passed to encoders, channel, and decoder.

        Returns:
            torch.Tensor: The output tensor from the joint decoder, typically containing
                the concatenated reconstructed messages for all users.
                Shape: (batch_size, num_users * decoded_message_dim_per_user).
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

        # 1. Encode messages for each user, passing *args and **kwargs
        encoded_signals = []
        for i in range(self.num_users):
            # Use the single shared encoder or the specific encoder for the user
            encoder_idx = 0 if self.shared_encoder else i
            # Ensure index is valid even if list length is 1 when shared
            if encoder_idx >= len(self.encoders):
                raise IndexError(f"Encoder index {encoder_idx} out of range for encoders list length {len(self.encoders)}.")
            encoder = self.encoders[encoder_idx]
            encoded_signals.append(encoder(x[i], *args, **kwargs))

        # 2. Combine encoded signals (summing them simulates superposition on the channel)
        combined_signal = torch.sum(torch.stack(encoded_signals), dim=0)

        # 3. Apply power constraint to the combined signal
        constrained_signal = self.power_constraint(combined_signal)

        # 4. Pass the combined signal through the channel
        # Pass *args and **kwargs to the channel
        received_signal = self.channel(constrained_signal, *args, **kwargs)

        # 5. Decode the received signal
        # Separate decoding using individual decoders
        reconstructed_signals = []
        for i in range(self.num_users):
            if i >= len(self.decoders):
                raise IndexError(f"Decoder index {i} out of range for decoders list length {len(self.decoders)}.")
            decoder = self.decoders[i]
            reconstructed_signals.append(decoder(received_signal, *args, **kwargs))
        # Concatenate the outputs along the feature dimension
        reconstructed_messages = torch.cat(reconstructed_signals, dim=1)

        # The joint decoder should output the concatenated messages
        return reconstructed_messages
