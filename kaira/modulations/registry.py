"""Modulation registry for Kaira."""

from typing import Callable, Dict, Literal, Optional, Type, Union

from .base import BaseDemodulator, BaseModulator


class ModulationRegistry:
    """A registry for modulations in Kaira.

    This class provides a centralized registry for all modulators and demodulators, making it
    easier to instantiate them by name with appropriate parameters.
    """

    _modulators: Dict[str, Type[BaseModulator]] = {}
    _demodulators: Dict[str, Type[BaseDemodulator]] = {}

    @classmethod
    def register(cls, name: str, modulation_class: Union[Type[BaseModulator], Type[BaseDemodulator]], mode: Literal["modulator", "demodulator"]) -> None:
        """Register a modulation class in the registry.

        Args:
            name (str): The name to register the modulation under.
            modulation_class (Union[Type[BaseModulator], Type[BaseDemodulator]]): The modulation class to register.
            mode (Literal["modulator", "demodulator"]): Whether the class is a modulator or demodulator.
        """
        if mode == "modulator":
            cls._modulators[name] = modulation_class
        elif mode == "demodulator":
            cls._demodulators[name] = modulation_class
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be either 'modulator' or 'demodulator'.")

    @classmethod
    def register_modulator(cls, name: Optional[str] = None) -> Callable:
        """Decorator to register a modulator class in the registry.

        Args:
            name (Optional[str], optional): The name to register the modulator under.
                                 If None, the class name will be used (converted to lowercase).

        Returns:
            callable: A decorator function that registers the modulator class.
        """

        def decorator(modulation_class):
            modulation_name = name if name is not None else modulation_class.__name__.lower()
            cls.register(modulation_name, modulation_class, mode="modulator")
            return modulation_class

        return decorator

    @classmethod
    def register_demodulator(cls, name: Optional[str] = None) -> Callable:
        """Decorator to register a demodulator class in the registry.

        Args:
            name (Optional[str], optional): The name to register the demodulator under.
                                 If None, the class name will be used (converted to lowercase).

        Returns:
            callable: A decorator function that registers the demodulator class.
        """

        def decorator(modulation_class):
            modulation_name = name if name is not None else modulation_class.__name__.lower()
            cls.register(modulation_name, modulation_class, mode="demodulator")
            return modulation_class

        return decorator

    @classmethod
    def get_modulator(cls, name: str) -> Type[BaseModulator]:
        """Get a modulator class by name.

        Args:
            name (str): The name of the modulator to get.

        Returns:
            Type[BaseModulator]: The modulator class.

        Raises:
            KeyError: If the modulator is not registered.
        """
        if name not in cls._modulators:
            raise KeyError(f"Modulator '{name}' not found in registry. Available modulators: {list(cls._modulators.keys())}")
        return cls._modulators[name]

    @classmethod
    def get_demodulator(cls, name: str) -> Type[BaseDemodulator]:
        """Get a demodulator class by name.

        Args:
            name (str): The name of the demodulator to get.

        Returns:
            Type[BaseDemodulator]: The demodulator class.

        Raises:
            KeyError: If the demodulator is not registered.
        """
        if name not in cls._demodulators:
            raise KeyError(f"Demodulator '{name}' not found in registry. Available demodulators: {list(cls._demodulators.keys())}")
        return cls._demodulators[name]

    @classmethod
    def get(cls, name: str, mode: Literal["modulator", "demodulator"] = "modulator") -> Union[Type[BaseModulator], Type[BaseDemodulator]]:
        """Get a modulation class by name.

        Args:
            name (str): The name of the modulation to get.
            mode (Literal["modulator", "demodulator"]): Whether to get a modulator or demodulator.

        Returns:
            Union[Type[BaseModulator], Type[BaseDemodulator]]: The modulation class.

        Raises:
            KeyError: If the modulation is not registered.
        """
        if mode == "modulator":
            return cls.get_modulator(name)
        elif mode == "demodulator":
            return cls.get_demodulator(name)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be either 'modulator' or 'demodulator'.")

    @classmethod
    def create_modulator(cls, name: str, **kwargs) -> BaseModulator:
        """Create a modulator instance by name.

        Args:
            name (str): The name of the modulator to create.
            **kwargs: Additional arguments to pass to the modulator constructor.

        Returns:
            BaseModulator: The instantiated modulator.
        """
        modulator_class = cls.get_modulator(name)
        return modulator_class(**kwargs)

    @classmethod
    def create_demodulator(cls, name: str, **kwargs) -> BaseDemodulator:
        """Create a demodulator instance by name.

        Args:
            name (str): The name of the demodulator to create.
            **kwargs: Additional arguments to pass to the demodulator constructor.

        Returns:
            BaseDemodulator: The instantiated demodulator.
        """
        demodulator_class = cls.get_demodulator(name)
        return demodulator_class(**kwargs)

    @classmethod
    def create(cls, name: str, mode: Literal["modulator", "demodulator"] = "modulator", **kwargs) -> Union[BaseModulator, BaseDemodulator]:
        """Create a modulation instance by name.

        Args:
            name (str): The name of the modulation to create.
            mode (Literal["modulator", "demodulator"]): Whether to create a modulator or demodulator.
            **kwargs: Additional arguments to pass to the modulation constructor.

        Returns:
            Union[BaseModulator, BaseDemodulator]: The instantiated modulation.
        """
        if mode == "modulator":
            return cls.create_modulator(name, **kwargs)
        elif mode == "demodulator":
            return cls.create_demodulator(name, **kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be either 'modulator' or 'demodulator'.")

    @classmethod
    def list_modulators(cls) -> list:
        """List all available modulators in the registry.

        Returns:
            list: A list of modulator names.
        """
        return list(cls._modulators.keys())

    @classmethod
    def list_demodulators(cls) -> list:
        """List all available demodulators in the registry.

        Returns:
            list: A list of demodulator names.
        """
        return list(cls._demodulators.keys())

    @classmethod
    def list_modulations(cls) -> dict:
        """List all available modulations in the registry.

        Returns:
            dict: A dictionary containing modulator and demodulator names.
        """
        return {
            "modulators": cls.list_modulators(),
            "demodulators": cls.list_demodulators(),
        }
