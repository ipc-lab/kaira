"""Comprehensive tests for the modulation registry."""

import pytest
import torch

from kaira.modulations.base import BaseDemodulator, BaseModulator
from kaira.modulations.dpsk import DPSKModulator
from kaira.modulations.psk import PSKModulator
from kaira.modulations.qam import QAMModulator
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

    def forward(self, x, noise_var=None):
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

    def test_get_method_routing(self):
        """Test that the get method correctly routes to get_modulator or get_demodulator."""
        # Test get with modulator mode
        modulator_class = ModulationRegistry.get("test_mod", mode="modulator")
        assert modulator_class == MockModulator
        assert modulator_class == ModulationRegistry.get_modulator("test_mod")

        # Test get with demodulator mode
        demodulator_class = ModulationRegistry.get("test_demod", mode="demodulator")
        assert demodulator_class == MockDemodulator
        assert demodulator_class == ModulationRegistry.get_demodulator("test_demod")


class TestModulationRegistryAllSchemes:
    """Test registry functionality with all modulation schemes."""

    def test_standard_scheme_registration(self):
        """Verify that standard modulation schemes are registered."""
        modulators = ModulationRegistry.list_modulators()
        demodulators = ModulationRegistry.list_demodulators()

        # Check that key modulation schemes are registered
        standard_schemes = ["pskmodulator", "qammodulator", "dpskmodulator", "identitymodulator"]
        for scheme in standard_schemes:
            assert scheme in modulators, f"Modulator {scheme} not registered"

        standard_demod_schemes = ["pskdemodulator", "qamdemodulator", "dpskdemodulator", "identitydemodulator"]
        for scheme in standard_demod_schemes:
            assert scheme in demodulators, f"Demodulator {scheme} not registered"

        # Check specific aliases too
        assert "bpskmodulator" in modulators, "BPSK modulator alias not registered"
        assert "qpskmodulator" in modulators, "QPSK modulator alias not registered"
        assert "dbpsk" in modulators, "DBPSK modulator alias not registered"
        assert "dqpsk" in modulators, "DQPSK modulator alias not registered"

    def test_create_by_specific_aliases(self):
        """Test creating modulators and demodulators by their aliases."""
        # Test creating BPSK
        bpsk_mod = ModulationRegistry.create("bpskmodulator", mode="modulator")
        # Different implementations might use either PSKModulator with order=2 or a custom BPSKModulator
        if isinstance(bpsk_mod, PSKModulator):
            assert bpsk_mod.order == 2
        else:
            # For custom BPSKModulator implementations, check bits_per_symbol
            assert bpsk_mod.bits_per_symbol == 1

        # Test creating QPSK
        qpsk_mod = ModulationRegistry.create("qpskmodulator", mode="modulator")
        if isinstance(qpsk_mod, PSKModulator):
            assert qpsk_mod.order == 4
        else:
            # For custom QPSKModulator implementations, check bits_per_symbol
            assert qpsk_mod.bits_per_symbol == 2

        # Test creating 16-QAM - specify order parameter
        qam16_mod = ModulationRegistry.create("qammodulator", mode="modulator", order=16)
        assert isinstance(qam16_mod, QAMModulator)
        assert qam16_mod.order == 16

        # Test creating DBPSK
        dbpsk_mod = ModulationRegistry.create("dbpsk", mode="modulator")
        assert isinstance(dbpsk_mod, DPSKModulator)
        assert dbpsk_mod.order == 2

    def test_modulator_demodulator_pairs(self):
        """Test that modulator-demodulator pairs can be created and are compatible."""
        modulation_pairs = [
            ("pskmodulator", "pskdemodulator", {"order": 4}),
            ("qammodulator", "qamdemodulator", {"order": 16}),
            ("dpskmodulator", "dpskdemodulator", {"order": 4}),
            ("identitymodulator", "identitydemodulator", {}),
            ("bpskmodulator", "bpskdemodulator", {}),
            ("qpskmodulator", "qpskdemodulator", {}),
        ]

        for mod_name, demod_name, params in modulation_pairs:
            # Create modulator and demodulator with required parameters
            modulator = ModulationRegistry.create(mod_name, mode="modulator", **params)
            demodulator = ModulationRegistry.create(demod_name, mode="demodulator", **params)

            # Check they have compatible bits_per_symbol
            assert modulator.bits_per_symbol == demodulator.bits_per_symbol, f"bits_per_symbol mismatch for {mod_name}/{demod_name}"

            # For non-differential schemes, test basic modulation/demodulation cycle
            if not isinstance(modulator, DPSKModulator):
                # Create test input
                n_bits = modulator.bits_per_symbol * 4
                bits = torch.randint(0, 2, (n_bits,), dtype=torch.float)

                # Complete modulation cycle
                symbols = modulator(bits)
                demodulator(symbols)

                # For PAM/PSK/QAM, the shapes should match
                if mod_name in ["bpskmodulator", "qpskmodulator"]:
                    # Check shapes (accounting for bit grouping)
                    expected_symbol_count = n_bits // modulator.bits_per_symbol
                    assert symbols.shape[0] == expected_symbol_count, f"Unexpected symbol count for {mod_name}"

    def test_parameter_passing(self):
        """Test that parameters are correctly passed to modulators/demodulators."""
        # Create PSK with non-default order
        psk8_mod = ModulationRegistry.create("pskmodulator", mode="modulator", order=8)
        assert isinstance(psk8_mod, PSKModulator)
        assert psk8_mod.order == 8
        assert psk8_mod.bits_per_symbol == 3  # log2(8)

        # Create QAM with normalization disabled
        qam16_mod = ModulationRegistry.create("qammodulator", mode="modulator", order=16, normalize=False)
        assert isinstance(qam16_mod, QAMModulator)
        assert qam16_mod.order == 16

        # Create DPSK with specific gray coding setting
        dpsk_mod = ModulationRegistry.create("dpskmodulator", mode="modulator", order=4, gray_coding=False)
        assert isinstance(dpsk_mod, DPSKModulator)
        assert dpsk_mod.order == 4
        assert dpsk_mod.gray_coding is False


# Test invalid use cases and edge cases
class TestModulationRegistryEdgeCases:
    """Test registry edge cases."""

    def test_register_duplicate(self):
        """Test registering a duplicate name."""
        # Save original registry
        original_modulators = ModulationRegistry._modulators.copy()

        try:
            # First registration
            ModulationRegistry.register("duplicate", MockModulator, mode="modulator")

            # Register the same name again - should overwrite without error
            ModulationRegistry.register("duplicate", PSKModulator, mode="modulator")

            # Check that it was overwritten
            assert ModulationRegistry._modulators["duplicate"] == PSKModulator
        finally:
            # Restore original registry
            ModulationRegistry._modulators = original_modulators

    def test_registration_with_inheritance(self):
        """Test registration with inheritance."""

        # Create a class hierarchy
        class BaseTestModulator(BaseModulator):
            def forward(self, x):
                return x

            @property
            def bits_per_symbol(self):
                return 1

        class DerivedTestModulator(BaseTestModulator):
            pass

        # Save original registry
        original_modulators = ModulationRegistry._modulators.copy()

        try:
            # Register the base class
            ModulationRegistry.register("base_test", BaseTestModulator, mode="modulator")

            # Register the derived class
            ModulationRegistry.register("derived_test", DerivedTestModulator, mode="modulator")

            # Both should be accessible
            assert ModulationRegistry._modulators["base_test"] == BaseTestModulator
            assert ModulationRegistry._modulators["derived_test"] == DerivedTestModulator

            # Create instances of both
            base_mod = ModulationRegistry.create("base_test", mode="modulator")
            derived_mod = ModulationRegistry.create("derived_test", mode="modulator")

            assert isinstance(base_mod, BaseTestModulator)
            assert isinstance(derived_mod, DerivedTestModulator)
            # Derived should also be instance of base
            assert isinstance(derived_mod, BaseTestModulator)
        finally:
            # Restore original registry
            ModulationRegistry._modulators = original_modulators

    def test_create_with_mode_and_helper_methods(self):
        """Test creating modulators/demodulators using different API methods."""
        # Using helper methods
        mod1 = ModulationRegistry.create_modulator("pskmodulator", order=4)

        # Using generic method with mode
        mod2 = ModulationRegistry.create("pskmodulator", mode="modulator", order=4)

        # Both should create the same type of object
        assert isinstance(mod1, PSKModulator)
        assert isinstance(mod2, PSKModulator)

        # With same parameters
        assert mod1.order == 4
        assert mod2.order == 4

    def test_invalid_parameters(self):
        """Test creating a modulator with invalid parameters."""
        # PSK requires order to be a power of 2
        with pytest.raises(ValueError):
            ModulationRegistry.create("pskmodulator", mode="modulator", order=3)

        # QAM requires order to be a perfect square and a power of 4
        with pytest.raises(ValueError):
            ModulationRegistry.create("qammodulator", mode="modulator", order=32)
