"""Comprehensive tests for the modulation registry."""
import pytest
import torch
from torch import nn

from kaira.modulations.base import BaseModulator, BaseDemodulator
from kaira.modulations.registry import ModulationRegistry


# Create mock modulation classes for testing
class MockModulator(BaseModulator):
    """Mock modulator for testing."""
    
    def __init__(self, order=4, test_param=None):
        super().__init__()
        self.order = order
        self.test_param = test_param
        # bits_per_symbol should be a property in the mock to avoid attribute error
    
    @property
    def bits_per_symbol(self):
        """Return bits per symbol."""
        return 2
    
    def forward(self, x):
        """Forward pass."""
        return x


class MockDemodulator(BaseDemodulator):
    """Mock demodulator for testing."""
    
    def __init__(self, order=4, test_param=None):
        super().__init__()
        self.order = order
        self.test_param = test_param
        # bits_per_symbol should be a property in the mock to avoid attribute error
    
    @property
    def bits_per_symbol(self):
        """Return bits per symbol."""
        return 2
    
    def forward(self, x):
        """Forward pass."""
        return x


# Test the registry functionality
class TestModulationRegistry:
    """Test the ModulationRegistry class."""
    
    def setup_method(self):
        """Setup method to clear registry before each test."""
        # Save original registry
        self.original_modulators = ModulationRegistry._modulators.copy()
        self.original_demodulators = ModulationRegistry._demodulators.copy()
        
        # Clear registry
        ModulationRegistry._modulators.clear()
        ModulationRegistry._demodulators.clear()
        
        # Register test modulations
        ModulationRegistry.register("test_mod", MockModulator, mode="modulator")
        ModulationRegistry.register("test_demod", MockDemodulator, mode="demodulator")
    
    def teardown_method(self):
        """Teardown method to restore registry after each test."""
        # Restore original registry
        ModulationRegistry._modulators = self.original_modulators
        ModulationRegistry._demodulators = self.original_demodulators
    
    def test_register_invalid_mode(self):
        """Test registering with an invalid mode."""
        with pytest.raises(ValueError, match="Invalid mode"):
            ModulationRegistry.register("invalid", MockModulator, mode="invalid")
    
    def test_register_modulator_decorator(self):
        """Test the register_modulator decorator."""
        # Test with explicit name
        @ModulationRegistry.register_modulator(name="decorated_mod")
        class DecoratedModulator(MockModulator):
            pass
        
        assert "decorated_mod" in ModulationRegistry._modulators
        assert ModulationRegistry._modulators["decorated_mod"] == DecoratedModulator
        
        # Test with implicit name
        @ModulationRegistry.register_modulator()
        class AnotherModulator(MockModulator):
            pass
        
        assert "anothermodulator" in ModulationRegistry._modulators
        assert ModulationRegistry._modulators["anothermodulator"] == AnotherModulator
    
    def test_register_demodulator_decorator(self):
        """Test the register_demodulator decorator."""
        # Test with explicit name
        @ModulationRegistry.register_demodulator(name="decorated_demod")
        class DecoratedDemodulator(MockDemodulator):
            pass
        
        assert "decorated_demod" in ModulationRegistry._demodulators
        assert ModulationRegistry._demodulators["decorated_demod"] == DecoratedDemodulator
        
        # Test with implicit name
        @ModulationRegistry.register_demodulator()
        class AnotherDemodulator(MockDemodulator):
            pass
        
        assert "anotherdemodulator" in ModulationRegistry._demodulators
        assert ModulationRegistry._demodulators["anotherdemodulator"] == AnotherDemodulator
    
    def test_get_modulator_nonexistent(self):
        """Test getting a non-existent modulator."""
        with pytest.raises(KeyError, match="not found in registry"):
            ModulationRegistry.get_modulator("nonexistent")
    
    def test_get_demodulator_nonexistent(self):
        """Test getting a non-existent demodulator."""
        with pytest.raises(KeyError, match="not found in registry"):
            ModulationRegistry.get_demodulator("nonexistent")
    
    def test_get_with_invalid_mode(self):
        """Test the get method with an invalid mode."""
        with pytest.raises(ValueError, match="Invalid mode"):
            ModulationRegistry.get("test_mod", mode="invalid")
    
    def test_create_modulator(self):
        """Test creating a modulator instance."""
        # Test with default parameters
        mod = ModulationRegistry.create_modulator("test_mod")
        assert isinstance(mod, MockModulator)
        assert mod.order == 4
        assert mod.test_param is None
        
        # Test with custom parameters
        mod = ModulationRegistry.create_modulator("test_mod", order=8, test_param="test")
        assert isinstance(mod, MockModulator)
        assert mod.order == 8
        assert mod.test_param == "test"
    
    def test_create_demodulator(self):
        """Test creating a demodulator instance."""
        # Test with default parameters
        demod = ModulationRegistry.create_demodulator("test_demod")
        assert isinstance(demod, MockDemodulator)
        assert demod.order == 4
        assert demod.test_param is None
        
        # Test with custom parameters
        demod = ModulationRegistry.create_demodulator("test_demod", order=8, test_param="test")
        assert isinstance(demod, MockDemodulator)
        assert demod.order == 8
        assert demod.test_param == "test"
    
    def test_create_with_invalid_mode(self):
        """Test the create method with an invalid mode."""
        with pytest.raises(ValueError, match="Invalid mode"):
            ModulationRegistry.create("test_mod", mode="invalid")
    
    def test_list_modulators(self):
        """Test listing all modulators."""
        modulators = ModulationRegistry.list_modulators()
        assert "test_mod" in modulators
    
    def test_list_demodulators(self):
        """Test listing all demodulators."""
        demodulators = ModulationRegistry.list_demodulators()
        assert "test_demod" in demodulators
    
    def test_list_modulations(self):
        """Test listing all modulations."""
        modulations = ModulationRegistry.list_modulations()
        assert "modulators" in modulations
        assert "demodulators" in modulations
        assert "test_mod" in modulations["modulators"]
        assert "test_demod" in modulations["demodulators"]