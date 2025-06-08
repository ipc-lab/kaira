"""Uplink Multiple Access Channel (MAC) implementations.

This module provides channel models for uplink communication scenarios where multiple users
transmit simultaneously to a single receiver. The UplinkMACChannel uses a composition pattern,
accepting existing channel implementations as parameters to model different channel conditions for
individual user transmissions.
"""

from typing import Any, Dict, List, Optional, Union

import torch

from .base import BaseChannel
from .registry import ChannelRegistry


@ChannelRegistry.register_channel()
class UplinkMACChannel(BaseChannel):
    """Uplink Multiple Access Channel (MAC) for modeling multi-user uplink communications.

    This channel models uplink communication scenarios where multiple users transmit
    simultaneously to a single receiver. The channel uses a composition pattern,
    accepting existing channel implementations (e.g., FlatFadingChannel, AWGNChannel)
    as parameters to model different channel conditions for individual user transmissions.

    The channel applies per-user channel effects, models inter-user interference,
    and combines the signals according to the MAC model. This enables realistic
    simulation of uplink scenarios with different channel conditions per user.

    Mathematical Model:
        For N users, the received signal is:
        y = Σᵢ₌₁ᴺ hᵢ(xᵢ) + interference + noise

        where hᵢ(xᵢ) is the channel response for user i's signal xᵢ.

    Args:
        user_channels (Union[BaseChannel, List[BaseChannel]]): Channel instances for each user.
            Can be a single channel to be shared among all users, or a list of channels
            (one per user).
        num_users (Optional[int]): Number of users. Required if user_channels is a single
            channel instance. Inferred from the list length if user_channels is a list.
        user_gains (Optional[Union[float, List[float]]]): Per-user channel gains.
            Can be a single gain applied to all users or a list of gains (one per user).
            Defaults to 1.0 for all users.
        interference_power (float): Power of inter-user interference. Defaults to 0.0.
        combine_method (str): Method for combining user signals. Options: 'sum', 'weighted_sum'.
            Defaults to 'sum'.

    Example:
        >>> # Using the same AWGN channel for all users
        >>> from kaira.channels import AWGNChannel, UplinkMACChannel
        >>> base_channel = AWGNChannel(avg_noise_power=0.1)
        >>> uplink_channel = UplinkMACChannel(
        ...     user_channels=base_channel,
        ...     num_users=3,
        ...     user_gains=[1.0, 0.8, 0.6]
        ... )

        >>> # Using different channels for each user
        >>> from kaira.channels import FlatFadingChannel, RayleighFadingChannel
        >>> user_channels = [
        ...     AWGNChannel(avg_noise_power=0.1),
        ...     FlatFadingChannel(fading_type="rayleigh", coherence_time=10, avg_noise_power=0.05),
        ...     RayleighFadingChannel(coherence_time=5, avg_noise_power=0.15)
        ... ]
        >>> uplink_channel = UplinkMACChannel(user_channels=user_channels)
    """

    def __init__(
        self,
        user_channels: Union[BaseChannel, List[BaseChannel]],
        num_users: Optional[int] = None,
        user_gains: Optional[Union[float, List[float]]] = None,
        interference_power: float = 0.0,
        combine_method: str = "sum",
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the UplinkMAC channel.

        Args:
            user_channels (Union[BaseChannel, List[BaseChannel]]): Channel instances for each user.
            num_users (Optional[int]): Number of users. Required if user_channels is a single channel.
            user_gains (Optional[Union[float, List[float]]]): Per-user channel gains.
            interference_power (float): Power of inter-user interference. Defaults to 0.0.
            combine_method (str): Method for combining user signals. Defaults to 'sum'.
            *args: Variable length argument list passed to the base class.
            **kwargs: Arbitrary keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        # Validate and set up user channels
        if isinstance(user_channels, list):
            if num_users is None:
                num_users = len(user_channels)
            elif num_users != len(user_channels):
                raise ValueError(f"Number of user channels ({len(user_channels)}) must match num_users ({num_users})")
            self.user_channels = user_channels
            self.shared_channel = False
        elif isinstance(user_channels, BaseChannel):
            if num_users is None:
                raise ValueError("num_users must be specified when using a shared channel")
            if num_users <= 0:
                raise ValueError("num_users must be positive")
            self.user_channels = [user_channels] * num_users
            self.shared_channel = True
        else:
            raise TypeError("user_channels must be a BaseChannel instance or a list of BaseChannel instances")

        self.num_users = num_users

        # Validate and set up user gains
        if user_gains is None:
            self.user_gains = torch.ones(self.num_users, dtype=torch.float32)
        elif isinstance(user_gains, (int, float)):
            self.user_gains = torch.full((self.num_users,), float(user_gains), dtype=torch.float32)
        elif isinstance(user_gains, list):
            if len(user_gains) != self.num_users:
                raise ValueError(f"Length of user_gains ({len(user_gains)}) must match num_users ({self.num_users})")
            self.user_gains = torch.tensor(user_gains, dtype=torch.float32)
        else:
            raise TypeError("user_gains must be a number or a list of numbers")

        # Validate interference power
        if interference_power < 0:
            raise ValueError("interference_power must be non-negative")
        self.interference_power = interference_power

        # Validate combine method
        valid_methods = ["sum", "weighted_sum"]
        if combine_method not in valid_methods:
            raise ValueError(f"combine_method must be one of {valid_methods}")
        self.combine_method = combine_method

    def forward(
        self,
        x: List[torch.Tensor],
        *args: Any,
        user_csi: Optional[List[torch.Tensor]] = None,
        user_noise: Optional[List[torch.Tensor]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply uplink MAC channel effects to user signals.

        Args:
            x (List[torch.Tensor]): List of input signals, one per user.
                Each tensor should have the same shape.
            *args: Additional positional arguments passed to individual channels.
            user_csi (Optional[List[torch.Tensor]]): Per-user channel state information.
                If provided, should be a list of CSI tensors (one per user).
            user_noise (Optional[List[torch.Tensor]]): Per-user noise tensors.
                If provided, should be a list of noise tensors (one per user).
            **kwargs: Additional keyword arguments passed to individual channels.

        Returns:
            torch.Tensor: Combined received signal after applying channel effects
                and inter-user interference.

        Raises:
            ValueError: If the number of input signals doesn't match num_users.
            ValueError: If user_csi or user_noise lists don't match num_users.
        """
        user_signals = x

        # Validate inputs
        if not isinstance(x, list):
            raise TypeError("user_signals must be a list of torch.Tensors")

        if len(user_signals) != self.num_users:
            raise ValueError(f"Expected {self.num_users} user signals, got {len(user_signals)}")

        if user_csi is not None and len(user_csi) != self.num_users:
            raise ValueError(f"Expected {self.num_users} user_csi, got {len(user_csi)}")

        if user_noise is not None and len(user_noise) != self.num_users:
            raise ValueError(f"Expected {self.num_users} user_noise, got {len(user_noise)}")

        # Validate that all user signals have the same shape
        reference_shape = user_signals[0].shape
        for i, signal in enumerate(user_signals[1:], 1):
            if signal.shape != reference_shape:
                raise ValueError(f"All user signals must have the same shape. " f"User 0: {reference_shape}, User {i}: {signal.shape}")

        # Process each user's signal through their respective channel
        processed_signals = []
        for i in range(self.num_users):
            channel = self.user_channels[i]
            signal = user_signals[i]
            gain = self.user_gains[i]

            # Prepare channel-specific arguments
            channel_kwargs = kwargs.copy()
            if user_csi is not None:
                channel_kwargs["csi"] = user_csi[i]
            if user_noise is not None:
                channel_kwargs["noise"] = user_noise[i]

            # Apply channel effects
            processed_signal = channel(signal, *args, **channel_kwargs)

            # Apply user-specific gain
            gain_value = gain.item() if hasattr(gain, "item") else gain
            if gain_value != 1.0:
                processed_signal = processed_signal * gain_value

            processed_signals.append(processed_signal)

        # Add inter-user interference if specified
        if self.interference_power > 0:
            processed_signals = self._add_interference(processed_signals)

        # Combine signals according to the specified method
        combined_signal = self._combine_signals(processed_signals)

        return combined_signal

    def _add_interference(self, processed_signals: List[torch.Tensor]) -> List[torch.Tensor]:
        """Add inter-user interference to processed signals.

        Args:
            processed_signals (List[torch.Tensor]): List of processed user signals.

        Returns:
            List[torch.Tensor]: Signals with added interference.
        """
        if self.interference_power <= 0:
            return processed_signals

        # Create interference signals
        interfered_signals = []
        for i, signal in enumerate(processed_signals):
            # Generate interference from other users
            interference = torch.zeros_like(signal)
            for j, other_signal in enumerate(processed_signals):
                if i != j:  # Don't add self-interference
                    # Add scaled version of other user's signal as interference
                    interference_scale = torch.sqrt(torch.tensor(self.interference_power, device=signal.device))
                    interference += other_signal * interference_scale / torch.sqrt(torch.tensor(self.num_users - 1, device=signal.device))

            interfered_signals.append(signal + interference)

        return interfered_signals

    def _combine_signals(self, signals: List[torch.Tensor]) -> torch.Tensor:
        """Combine processed user signals according to the specified method.

        Args:
            signals (List[torch.Tensor]): List of processed user signals.

        Returns:
            torch.Tensor: Combined signal.
        """
        if self.combine_method == "sum":
            # Simple summation (superposition principle)
            return torch.sum(torch.stack(signals), dim=0)
        elif self.combine_method == "weighted_sum":
            # Weighted summation using user gains (gains already applied in forward method)
            return torch.sum(torch.stack(signals), dim=0)
        else:
            # This should not happen due to validation in __init__
            raise ValueError(f"Unknown combine method: {self.combine_method}")

    def get_user_csi(self, user_idx: int) -> Optional[torch.Tensor]:
        """Get channel state information for a specific user.

        Args:
            user_idx (int): Index of the user (0-based).

        Returns:
            Optional[torch.Tensor]: CSI for the specified user, if available.

        Raises:
            ValueError: If user_idx is out of range.
        """
        if not 0 <= user_idx < self.num_users:
            raise ValueError(f"User index {user_idx} is out of range for {self.num_users} users")

        channel = self.user_channels[user_idx]
        # Try to get CSI if the channel supports it
        if hasattr(channel, "get_csi"):
            return channel.get_csi()
        elif hasattr(channel, "csi"):
            return channel.csi
        else:
            return None

    def update_user_gain(self, user_idx: int, new_gain: float) -> None:
        """Update the channel gain for a specific user.

        Args:
            user_idx (int): Index of the user (0-based).
            new_gain (float): New gain value.

        Raises:
            ValueError: If user_idx is out of range.
        """
        if not 0 <= user_idx < self.num_users:
            raise ValueError(f"User index {user_idx} is out of range for {self.num_users} users")

        self.user_gains[user_idx] = float(new_gain)

    def update_interference_power(self, new_power: float) -> None:
        """Update the inter-user interference power.

        Args:
            new_power (float): New interference power.

        Raises:
            ValueError: If new_power is negative.
        """
        if new_power < 0:
            raise ValueError("interference_power must be non-negative")
        self.interference_power = new_power

    def get_config(self) -> Dict[str, Any]:
        """Get a dictionary of the channel's configuration.

        Returns:
            Dict[str, Any]: Dictionary of parameter names and values.
        """
        config = super().get_config()
        config.update(
            {
                "num_users": self.num_users,
                "user_gains": self.user_gains,
                "interference_power": self.interference_power,
                "combine_method": self.combine_method,
                "shared_channel": self.shared_channel,
            }
        )

        # Add channel configurations
        if self.shared_channel:
            config["shared_channel_config"] = self.user_channels[0].get_config()
        else:
            config["user_channel_configs"] = [ch.get_config() for ch in self.user_channels]

        return config

    def __repr__(self) -> str:
        """String representation of the UplinkMACChannel.

        Returns:
            str: String representation of the channel.
        """
        return f"UplinkMACChannel(num_users={self.num_users}, " f"user_gains={self.user_gains.tolist()}, " f"interference_power={self.interference_power}, " f"combine_method={self.combine_method})"
