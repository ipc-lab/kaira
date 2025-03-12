"""Channel registry for Kaira."""

from typing import Callable, Dict, Optional, Type

from .base import BaseChannel


class ChannelRegistry:
    """A registry for channels in Kaira.

    This class provides a centralized registry for all channels, making it easier to instantiate
    them by name with appropriate parameters.
    """

    _channels: Dict[str, Type[BaseChannel]] = {}

    @classmethod
    def register(cls, name: str, channel_class: Type[BaseChannel]) -> None:
        """Register a new channel in the registry.

        Args:
            name (str): The name to register the channel under.
            channel_class (Type[BaseChannel]): The channel class to register.
        """
        cls._channels[name] = channel_class

    @classmethod
    def register_channel(cls, name: Optional[str] = None) -> Callable:
        """Decorator to register a channel class in the registry.

        Args:
            name (Optional[str], optional): The name to register the channel under.
                                 If None, the class name will be used (converted to lowercase).

        Returns:
            callable: A decorator function that registers the channel class.
        """

        def decorator(channel_class):
            channel_name = name if name is not None else channel_class.__name__.lower()
            cls.register(channel_name, channel_class)
            return channel_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseChannel]:
        """Get a channel class by name.

        Args:
            name (str): The name of the channel to get.

        Returns:
            Type[BaseChannel]: The channel class.

        Raises:
            KeyError: If the channel is not registered.
        """
        if name not in cls._channels:
            raise KeyError(f"Channel '{name}' not found in registry. Available channels: {list(cls._channels.keys())}")
        return cls._channels[name]

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseChannel:
        """Create a channel instance by name.

        Args:
            name (str): The name of the channel to create.
            **kwargs: Additional arguments to pass to the channel constructor.

        Returns:
            BaseChannel: The instantiated channel.
        """
        channel_class = cls.get(name)
        return channel_class(**kwargs)

    @classmethod
    def list_channels(cls) -> list:
        """List all available channels in the registry.

        Returns:
            list: A list of channel names.
        """
        return list(cls._channels.keys())
