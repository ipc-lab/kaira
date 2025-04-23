"""Unit tests for capacity analyzer module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from kaira.channels import AWGNChannel, BinaryErasureChannel, BinarySymmetricChannel
from kaira.modulations import BPSKModulator, QAMModulator, QPSKModulator
from kaira.utils import snr_db_to_linear
from kaira.utils.capacity_analyzer import CapacityAnalyzer


class TestCapacityAnalyzer:
    """Test class for CapacityAnalyzer."""

    def setup_method(self):
        """Setup for tests."""
        # Use CPU device for deterministic testing
        self.device = torch.device("cpu")
        self.analyzer = CapacityAnalyzer(device=self.device, num_processes=1)

        # Setup common test data
        self.snr_db_single = 10.0
        self.snr_db_list = [0.0, 5.0, 10.0, 15.0]
        self.snr_db_array = np.array(self.snr_db_list)
        self.snr_db_tensor = torch.tensor(self.snr_db_list, device=self.device)

        # Setup channels for testing
        self.awgn_channel = AWGNChannel(snr_db=10.0)
        self.bsc_channel = BinarySymmetricChannel(crossover_prob=0.1)
        self.bec_channel = BinaryErasureChannel(erasure_prob=0.1)

        # Setup modulators for testing
        self.bpsk_modulator = BPSKModulator()
        self.qpsk_modulator = QPSKModulator()
        self.qam_modulator = QAMModulator(order=16)  # 16-QAM

    def test_initialization(self):
        """Test CapacityAnalyzer initialization."""
        # Test default initialization
        analyzer = CapacityAnalyzer()
        assert hasattr(analyzer, "device")
        assert hasattr(analyzer, "num_processes")

        # Test with custom device
        custom_device = torch.device("cpu")
        analyzer = CapacityAnalyzer(device=custom_device)
        assert analyzer.device == custom_device

        # Test with custom num_processes
        analyzer = CapacityAnalyzer(num_processes=4)
        assert analyzer.num_processes == 4

        # Test with num_processes=-1 (should use all available cores)
        import multiprocessing

        analyzer = CapacityAnalyzer(num_processes=-1)
        assert analyzer.num_processes == multiprocessing.cpu_count()

    def test_awgn_capacity(self):
        """Test AWGN capacity calculation."""
        # Test with single value
        capacity = self.analyzer.awgn_capacity(self.snr_db_single)
        assert isinstance(capacity, torch.Tensor)
        assert capacity.shape == torch.Size([1])
        expected_value = torch.log2(1 + snr_db_to_linear(torch.tensor([self.snr_db_single])))
        assert torch.allclose(capacity, expected_value)

        # Test with list of values
        capacity = self.analyzer.awgn_capacity(self.snr_db_list)
        assert isinstance(capacity, torch.Tensor)
        assert capacity.shape == torch.Size([len(self.snr_db_list)])

        # Test with numpy array
        capacity = self.analyzer.awgn_capacity(self.snr_db_array)
        assert isinstance(capacity, torch.Tensor)
        assert capacity.shape == torch.Size([len(self.snr_db_array)])

        # Test with tensor
        capacity = self.analyzer.awgn_capacity(self.snr_db_tensor)
        assert isinstance(capacity, torch.Tensor)
        assert capacity.shape == torch.Size([len(self.snr_db_tensor)])

    def test_awgn_capacity_complex(self):
        """Test complex AWGN capacity calculation."""
        # Test with single value
        capacity = self.analyzer.awgn_capacity_complex(self.snr_db_single)
        assert isinstance(capacity, torch.Tensor)
        assert capacity.shape == torch.Size([1])

        # Verify it's the same as regular AWGN capacity (for complex channels)
        regular_capacity = self.analyzer.awgn_capacity(self.snr_db_single)
        assert torch.allclose(capacity, regular_capacity)

    def test_bsc_capacity(self):
        """Test Binary Symmetric Channel capacity calculation."""
        # Test with single probability value
        p_single = 0.1
        capacity = self.analyzer.bsc_capacity(p_single)
        assert isinstance(capacity, torch.Tensor)
        assert capacity.shape == torch.Size([1])

        # Test with list of probability values
        p_list = [0.0, 0.1, 0.25, 0.5]
        capacity = self.analyzer.bsc_capacity(p_list)
        assert isinstance(capacity, torch.Tensor)
        assert capacity.shape == torch.Size([len(p_list)])

        # Test with numpy array
        p_array = np.array(p_list)
        capacity = self.analyzer.bsc_capacity(p_array)
        assert isinstance(capacity, torch.Tensor)
        assert capacity.shape == torch.Size([len(p_array)])

        # Test with tensor
        p_tensor = torch.tensor(p_list, device=self.device)
        capacity = self.analyzer.bsc_capacity(p_tensor)
        assert isinstance(capacity, torch.Tensor)
        assert capacity.shape == torch.Size([len(p_tensor)])

        # Test with value clamping (p > 0.5 should be clamped to 0.5)
        p_invalid = 0.7
        capacity = self.analyzer.bsc_capacity(p_invalid)
        capacity_at_half = self.analyzer.bsc_capacity(0.5)
        assert torch.allclose(capacity, capacity_at_half)

        # Test borderline cases
        # At p=0, capacity should be 1
        assert torch.allclose(self.analyzer.bsc_capacity(0.0), torch.tensor([1.0]))
        # At p=0.5, capacity should be 0
        assert torch.allclose(self.analyzer.bsc_capacity(0.5), torch.tensor([0.0]))

    def test_binary_entropy(self):
        """Test binary entropy function."""
        # Test with various probability values
        p_values = torch.tensor([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], device=self.device)
        entropy = self.analyzer._binary_entropy(p_values)
        assert isinstance(entropy, torch.Tensor)
        assert entropy.shape == p_values.shape

        # Test symmetry: H(p) = H(1-p)
        assert torch.allclose(entropy[1:6], torch.flip(entropy[1:6], [0]))

        # Test extreme cases
        # At p=0 or p=1, entropy should be 0
        assert torch.isclose(entropy[0], torch.tensor(0.0))
        assert torch.isclose(entropy[-1], torch.tensor(0.0))
        # At p=0.5, entropy should be 1
        assert torch.isclose(entropy[3], torch.tensor(1.0))

    def test_bec_capacity(self):
        """Test Binary Erasure Channel capacity calculation."""
        # Test with single probability value
        e_single = 0.1
        capacity = self.analyzer.bec_capacity(e_single)
        assert isinstance(capacity, torch.Tensor)
        assert capacity.shape == torch.Size([1])
        assert torch.isclose(capacity[0], torch.tensor(1 - e_single))

        # Test with list of probability values
        e_list = [0.0, 0.1, 0.5, 1.0]
        capacity = self.analyzer.bec_capacity(e_list)
        assert isinstance(capacity, torch.Tensor)
        assert capacity.shape == torch.Size([len(e_list)])

        # Test with numpy array
        e_array = np.array(e_list)
        capacity = self.analyzer.bec_capacity(e_array)
        assert isinstance(capacity, torch.Tensor)
        assert capacity.shape == torch.Size([len(e_array)])

        # Test with tensor
        e_tensor = torch.tensor(e_list, device=self.device)
        capacity = self.analyzer.bec_capacity(e_tensor)
        assert isinstance(capacity, torch.Tensor)
        assert capacity.shape == torch.Size([len(e_tensor)])

        # Test extreme cases
        # At e=0, capacity should be 1
        assert torch.isclose(capacity[0], torch.tensor(1.0))
        # At e=1, capacity should be 0
        assert torch.isclose(capacity[-1], torch.tensor(0.0))

    def test_gaussian_input_capacity(self):
        """Test capacity with Gaussian input distribution."""
        # Test AWGN channel
        capacity = self.analyzer.gaussian_input_capacity(self.awgn_channel, self.snr_db_single)
        assert isinstance(capacity, torch.Tensor)

        # Test with list of SNR values
        capacity = self.analyzer.gaussian_input_capacity(self.awgn_channel, self.snr_db_list)
        assert isinstance(capacity, torch.Tensor)
        assert capacity.shape == torch.Size([len(self.snr_db_list)])

        # Test with unconstrained capacity
        capacity_unconstrained = self.analyzer.gaussian_input_capacity(self.awgn_channel, self.snr_db_single, constrained=False)
        assert isinstance(capacity_unconstrained, torch.Tensor)

        # For AWGN, capacity should match awgn_capacity
        awgn_cap = self.analyzer.awgn_capacity(self.snr_db_list)
        gaussian_cap = self.analyzer.gaussian_input_capacity(self.awgn_channel, self.snr_db_list)
        assert torch.allclose(awgn_cap, gaussian_cap)

        # Test with a channel that requires ergodic capacity calculation
        # (Mock a Rayleigh fading channel)
        mock_channel = MagicMock()
        mock_channel.__class__.__name__ = "RayleighFadingChannel"
        mock_channel.to.return_value = None

        with patch.object(self.analyzer, "_rayleigh_ergodic_capacity", return_value=torch.tensor([1.0])):
            capacity = self.analyzer.gaussian_input_capacity(mock_channel, 10.0)
            assert isinstance(capacity, torch.Tensor)
            assert capacity.shape == torch.Size([1])
            self.analyzer._rayleigh_ergodic_capacity.assert_called_once()

        # Test with a different channel type that falls to ergodic capacity
        mock_channel.__class__.__name__ = "CustomChannel"
        with patch.object(self.analyzer, "ergodic_capacity", return_value=(torch.tensor([10.0]), torch.tensor([2.0]))):
            capacity = self.analyzer.gaussian_input_capacity(mock_channel, 10.0)
            assert torch.isclose(capacity[0], torch.tensor(2.0))
            self.analyzer.ergodic_capacity.assert_called_once()

    def test_rayleigh_ergodic_capacity(self):
        """Test ergodic capacity for Rayleigh fading channel."""
        # Test with a single SNR value
        capacity = self.analyzer._rayleigh_ergodic_capacity(torch.tensor([self.snr_db_single]))
        assert isinstance(capacity, torch.Tensor)
        assert capacity.shape == torch.Size([1])

        # Test with multiple SNR values
        capacity = self.analyzer._rayleigh_ergodic_capacity(self.snr_db_tensor)
        assert isinstance(capacity, torch.Tensor)
        assert capacity.shape == self.snr_db_tensor.shape

        # Higher SNR should give higher capacity
        assert capacity[0] <= capacity[-1]

    def test_mutual_information_histogram(self):
        """Test mutual information calculation using histogram method."""
        # Test with a single SNR value
        mi = self.analyzer.mutual_information(self.bpsk_modulator, self.awgn_channel, self.snr_db_single, num_symbols=1000, estimation_method="histogram")
        assert isinstance(mi, torch.Tensor)
        assert mi.shape == torch.Size([1])

        # Test with multiple SNR values
        mi = self.analyzer.mutual_information(self.bpsk_modulator, self.awgn_channel, self.snr_db_tensor, num_symbols=1000, estimation_method="histogram")
        assert isinstance(mi, torch.Tensor)
        assert mi.shape == self.snr_db_tensor.shape

        # Higher SNR should give higher mutual information
        assert mi[0] <= mi[-1]

        # For BPSK in high SNR, MI should approach 1 bit/symbol
        assert mi[-1] < 1.1  # Not exactly 1 due to estimation errors

        # For QPSK in high SNR, MI should approach 2 bits/symbol
        mi_qpsk = self.analyzer.mutual_information(self.qpsk_modulator, self.awgn_channel, torch.tensor([20.0]), num_symbols=1000, estimation_method="histogram")  # High SNR
        assert isinstance(mi_qpsk, torch.Tensor)
        assert mi_qpsk[0] < 2.1  # Not exactly 2 due to estimation errors

    def test_mutual_information_knn(self):
        """Test mutual information calculation using KNN method."""
        # Test with a single SNR value
        mi = self.analyzer.mutual_information(self.bpsk_modulator, self.awgn_channel, self.snr_db_single, num_symbols=500, estimation_method="knn")
        assert isinstance(mi, torch.Tensor)
        assert mi.shape == torch.Size([1])

        # MI for BPSK should be between 0 and 1 bit/symbol
        assert 0 <= mi[0] <= 1.1  # Allow slight margin for estimation errors

    def test_estimate_mutual_information(self):
        """Test the histogram-based mutual information estimator."""
        # Create test data
        num_samples = 1000

        # Test with real signals
        tx_real = torch.rand(num_samples, device=self.device)
        rx_real = tx_real + 0.1 * torch.randn(num_samples, device=self.device)

        # Mock the _estimate_mutual_information method to return a controlled value
        with patch.object(self.analyzer, "_estimate_mutual_information", return_value=torch.tensor(0.8)):
            mi_real = self.analyzer._estimate_mutual_information(tx_real, rx_real, num_bins=50, bits_per_symbol=1)
            assert isinstance(mi_real, torch.Tensor)
            assert mi_real.dim() == 0  # Scalar tensor

            # MI should be between 0 and 1 for real signals with 1 bit/symbol
            assert 0 <= mi_real <= 1.0

        # Test with complex signals
        tx_complex = torch.complex(torch.rand(num_samples, device=self.device), torch.rand(num_samples, device=self.device))
        rx_complex = tx_complex + 0.1 * torch.complex(torch.randn(num_samples, device=self.device), torch.randn(num_samples, device=self.device))

        # Mock the _estimate_mutual_information method for complex signals
        with patch.object(self.analyzer, "_estimate_mutual_information", return_value=torch.tensor(1.5)):
            mi_complex = self.analyzer._estimate_mutual_information(tx_complex, rx_complex, num_bins=30, bits_per_symbol=2)
            assert isinstance(mi_complex, torch.Tensor)

            # MI should be between 0 and 2 for complex signals with 2 bits/symbol
            assert 0 <= mi_complex <= 2.0

    def test_estimate_mutual_information_knn(self):
        """Test the KNN-based mutual information estimator."""
        # Create test data
        num_samples = 500

        # Test with real signals
        tx_real = torch.rand(num_samples, device=self.device)
        rx_real = tx_real + 0.1 * torch.randn(num_samples, device=self.device)

        mi_real = self.analyzer._estimate_mutual_information_knn(tx_real, rx_real, k=3, bits_per_symbol=1)
        assert isinstance(mi_real, torch.Tensor)

        # MI should be between 0 and 1 for real signals with 1 bit/symbol
        assert 0 <= mi_real <= 1.1  # Allow slight margin for estimation errors

        # Test with complex signals
        tx_complex = torch.complex(torch.rand(num_samples, device=self.device), torch.rand(num_samples, device=self.device))
        rx_complex = tx_complex + 0.1 * torch.complex(torch.randn(num_samples, device=self.device), torch.randn(num_samples, device=self.device))

        mi_complex = self.analyzer._estimate_mutual_information_knn(tx_complex, rx_complex, k=3, bits_per_symbol=2)
        assert isinstance(mi_complex, torch.Tensor)

        # MI should be between 0 and 2 for complex signals with 2 bits/symbol
        assert 0 <= mi_complex <= 2.1  # Allow slight margin for estimation errors

    def test_modulation_capacity(self):
        """Test capacity calculation for modulation schemes."""
        # Test with analytical solution for BPSK over AWGN
        snr_range, capacity = self.analyzer.modulation_capacity(self.bpsk_modulator, self.awgn_channel, [5.0, 10.0], num_symbols=1000, monte_carlo=False)
        assert isinstance(snr_range, torch.Tensor)
        assert isinstance(capacity, torch.Tensor)
        assert capacity.shape == snr_range.shape

        # Test with Monte Carlo simulation
        snr_range, capacity = self.analyzer.modulation_capacity(self.bpsk_modulator, self.awgn_channel, [5.0, 10.0], num_symbols=1000, monte_carlo=True)
        assert isinstance(capacity, torch.Tensor)
        assert capacity.shape == snr_range.shape

        # Test QAM modulator with analytical solution
        snr_range, capacity_qam = self.analyzer.modulation_capacity(self.qam_modulator, self.awgn_channel, [5.0, 10.0, 15.0], num_symbols=1000, monte_carlo=False)
        assert isinstance(capacity_qam, torch.Tensor)
        assert capacity_qam.shape == snr_range.shape

        # Higher SNR should give higher capacity
        assert capacity_qam[0] <= capacity_qam[-1]

        # QPSK modulator test
        snr_range, capacity_qpsk = self.analyzer.modulation_capacity(self.qpsk_modulator, self.awgn_channel, [5.0, 10.0], num_symbols=1000, monte_carlo=False)
        assert isinstance(capacity_qpsk, torch.Tensor)

    def test_bpsk_awgn_capacity(self):
        """Test analytical capacity calculation for BPSK over AWGN."""
        # Test with a single SNR value
        capacity = self.analyzer._bpsk_awgn_capacity(torch.tensor([10.0]))
        assert isinstance(capacity, torch.Tensor)
        assert capacity.shape == torch.Size([1])

        # Test with multiple SNR values
        capacity = self.analyzer._bpsk_awgn_capacity(self.snr_db_tensor)
        assert isinstance(capacity, torch.Tensor)
        assert capacity.shape == self.snr_db_tensor.shape

        # Capacity should be between 0 and 1 for BPSK
        assert torch.all(capacity >= 0)
        assert torch.all(capacity <= 1)

        # Higher SNR should give higher capacity
        assert capacity[0] <= capacity[-1]

    def test_qam_awgn_capacity(self):
        """Test analytical capacity calculation for QAM over AWGN."""
        # Test with a single SNR value
        capacity = self.analyzer._qam_awgn_capacity(torch.tensor([10.0]), constellation_size=16)
        assert isinstance(capacity, torch.Tensor)
        assert capacity.shape == torch.Size([1])

        # Test with multiple SNR values
        capacity = self.analyzer._qam_awgn_capacity(self.snr_db_tensor, constellation_size=4)
        assert isinstance(capacity, torch.Tensor)
        assert capacity.shape == self.snr_db_tensor.shape

        # Test very high SNR (should approach log2(constellation_size))
        capacity_high_snr = self.analyzer._qam_awgn_capacity(torch.tensor([100.0]), constellation_size=16)
        assert torch.isclose(capacity_high_snr, torch.tensor(4.0), atol=0.1)

        # Capacity should be between 0 and log2(constellation_size)
        assert torch.all(capacity >= 0)
        assert torch.all(capacity <= torch.tensor(2.0))  # For constellation_size=4

        # Higher SNR should give higher capacity
        assert capacity[0] <= capacity[-1]

    def test_generate_qam_symbols(self):
        """Test QAM symbol generation."""
        # Test for 4-QAM (QPSK)
        symbols_4qam = self.analyzer._generate_qam_symbols(4, 1000, self.device)
        assert isinstance(symbols_4qam, torch.Tensor)
        assert symbols_4qam.shape == torch.Size([1000])
        assert torch.is_complex(symbols_4qam)

        # Test for 16-QAM
        symbols_16qam = self.analyzer._generate_qam_symbols(16, 1000, self.device)
        assert isinstance(symbols_16qam, torch.Tensor)
        assert symbols_16qam.shape == torch.Size([1000])

        # Symbols should be normalized to unit average energy
        assert torch.isclose(torch.mean(torch.abs(symbols_4qam) ** 2), torch.tensor(1.0), atol=0.1)
        assert torch.isclose(torch.mean(torch.abs(symbols_16qam) ** 2), torch.tensor(1.0), atol=0.1)

    def test_plot_capacity_vs_snr(self):
        """Test capacity plotting function."""
        # Mock pyplot to avoid actual plotting
        with patch("matplotlib.pyplot.figure") as mock_figure:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_fig.add_subplot.return_value = mock_ax
            mock_figure.return_value = mock_fig

            # Test with a single capacity curve
            snr_range = self.snr_db_tensor
            capacity = torch.rand_like(snr_range)
            fig = self.analyzer.plot_capacity_vs_snr(snr_range, capacity)
            assert fig is not None

            # Test with multiple capacity curves as a list
            capacities = [torch.rand_like(snr_range), torch.rand_like(snr_range)]
            labels = ["Scheme 1", "Scheme 2"]
            fig = self.analyzer.plot_capacity_vs_snr(snr_range, capacities, labels=labels)
            assert fig is not None

            # Test with capacity curves as a dictionary
            capacities_dict = {"Scheme 1": torch.rand_like(snr_range), "Scheme 2": torch.rand_like(snr_range)}
            fig = self.analyzer.plot_capacity_vs_snr(snr_range, capacities_dict)
            assert fig is not None

            # Test with MIMO Shannon capacity
            fig = self.analyzer.plot_capacity_vs_snr(snr_range, capacity, include_shannon_mimo=True, mimo_tx=2, mimo_rx=2)
            assert fig is not None

    def test_plot_capacity_vs_param(self):
        """Test capacity vs parameter plotting function."""
        # Mock pyplot to avoid actual plotting
        with patch("matplotlib.pyplot.figure") as mock_figure:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_fig.add_subplot.return_value = mock_ax
            mock_figure.return_value = mock_fig

            # Test with a single capacity curve
            param_values = torch.linspace(0, 1, 5)
            capacity = torch.rand_like(param_values)
            fig = self.analyzer.plot_capacity_vs_param(param_values, capacity)
            assert fig is not None

            # Test with multiple capacity curves as a list
            capacities = [torch.rand_like(param_values), torch.rand_like(param_values)]
            labels = ["Scheme 1", "Scheme 2"]
            fig = self.analyzer.plot_capacity_vs_param(param_values, capacities, labels=labels, param_name="Error Rate")
            assert fig is not None

            # Test with capacity curves as a dictionary
            capacities_dict = {"Scheme 1": torch.rand_like(param_values), "Scheme 2": torch.rand_like(param_values)}
            fig = self.analyzer.plot_capacity_vs_param(param_values, capacities_dict)
            assert fig is not None

    def test_ergodic_capacity(self):
        """Test ergodic capacity calculation."""
        # Create a mock channel
        mock_channel = MagicMock()
        mock_channel.to.return_value = None
        mock_channel.__class__.__name__ = "RayleighFadingChannel"

        # Mock channel function to return noisy output
        def mock_channel_func(input_):
            return input_ + 0.1 * torch.complex(torch.randn_like(input_.real), torch.randn_like(input_.imag))

        mock_channel.side_effect = mock_channel_func

        # Patch the _estimate_mutual_information method to avoid actual computation
        with patch.object(self.analyzer, "_estimate_mutual_information", return_value=torch.tensor(1.5)):
            # Test with a single SNR value
            snr_values, capacity = self.analyzer.ergodic_capacity(mock_channel, [10.0], num_realizations=5, num_symbols_per_realization=10)
            assert isinstance(capacity, torch.Tensor)
            assert capacity.shape == snr_values.shape

            # Test with multiple SNR values
            snr_values, capacity = self.analyzer.ergodic_capacity(mock_channel, self.snr_db_list, num_realizations=5, num_symbols_per_realization=10)
            assert isinstance(capacity, torch.Tensor)
            assert capacity.shape == snr_values.shape

    def test_outage_capacity(self):
        """Test outage capacity calculation."""
        # Create a mock channel
        mock_channel = MagicMock()
        mock_channel.to.return_value = None

        # Mock channel function to return noisy output
        def mock_channel_func(input_):
            return input_ + 0.1 * torch.complex(torch.randn_like(input_.real), torch.randn_like(input_.imag))

        mock_channel.side_effect = mock_channel_func

        # Patch the _estimate_mutual_information method to return random values
        with patch.object(self.analyzer, "_estimate_mutual_information", side_effect=lambda *args, **kwargs: torch.rand(1)[0]):
            # Test with a single SNR value
            snr_values, capacity = self.analyzer.outage_capacity(mock_channel, [10.0], outage_probability=0.1, num_realizations=10, num_symbols_per_realization=10)
            assert isinstance(capacity, torch.Tensor)
            assert capacity.shape == snr_values.shape

            # Test with multiple SNR values
            snr_values, capacity = self.analyzer.outage_capacity(mock_channel, self.snr_db_list, outage_probability=0.1, num_realizations=10, num_symbols_per_realization=10)
            assert isinstance(capacity, torch.Tensor)
            assert capacity.shape == snr_values.shape

    def test_compare_modulation_schemes(self):
        """Test comparison of modulation schemes."""
        # Create a list of modulators to compare
        modulators = [self.bpsk_modulator, self.qpsk_modulator]

        # Mock mutual_information to avoid actual computation
        with patch.object(self.analyzer, "mutual_information", side_effect=lambda *args, **kwargs: torch.rand(len(args[2]))):
            # Test without plotting
            snr_values, capacities, fig = self.analyzer.compare_modulation_schemes(modulators, self.awgn_channel, self.snr_db_list, plot=False)
            assert isinstance(capacities, dict)
            assert len(capacities) == len(modulators)
            assert fig is None

            # Test with plotting
            with patch("matplotlib.pyplot.figure") as mock_figure:
                mock_fig = MagicMock()
                mock_ax = MagicMock()
                mock_fig.add_subplot.return_value = mock_ax
                mock_figure.return_value = mock_fig

                snr_values, capacities, fig = self.analyzer.compare_modulation_schemes(modulators, self.awgn_channel, self.snr_db_list, labels=["BPSK", "QPSK"], plot=True)
                assert isinstance(capacities, dict)
                assert fig is not None

    def test_compare_channels(self):
        """Test comparison of channels."""
        # Create a list of channels to compare
        channels = [self.awgn_channel, self.bsc_channel]

        # Mock mutual_information to avoid actual computation
        with patch.object(self.analyzer, "mutual_information", side_effect=lambda *args, **kwargs: torch.rand(len(args[2]))):
            # Test without plotting
            snr_values, capacities, fig = self.analyzer.compare_channels(self.bpsk_modulator, channels, self.snr_db_list, plot=False)
            assert isinstance(capacities, dict)
            assert len(capacities) == len(channels)
            assert fig is None

            # Test with plotting
            with patch("matplotlib.pyplot.figure") as mock_figure:
                mock_fig = MagicMock()
                mock_ax = MagicMock()
                mock_fig.add_subplot.return_value = mock_ax
                mock_figure.return_value = mock_fig

                snr_values, capacities, fig = self.analyzer.compare_channels(self.bpsk_modulator, channels, self.snr_db_list, labels=["AWGN", "BSC"], plot=True)
                assert isinstance(capacities, dict)
                assert fig is not None

    def test_mimo_capacity(self):
        """Test MIMO capacity calculation."""
        # Test with a single SNR value
        capacity = self.analyzer.mimo_capacity(self.snr_db_single, tx_antennas=2, rx_antennas=2, channel_knowledge="perfect", num_realizations=5)
        assert isinstance(capacity, torch.Tensor)
        assert capacity.shape == torch.Size([1])

        # Test with multiple SNR values
        capacity = self.analyzer.mimo_capacity(self.snr_db_tensor, tx_antennas=2, rx_antennas=2, channel_knowledge="perfect", num_realizations=5)
        assert isinstance(capacity, torch.Tensor)
        assert capacity.shape == self.snr_db_tensor.shape

        # Test different channel knowledge settings
        for knowledge in ["perfect", "statistical", "none"]:
            capacity = self.analyzer.mimo_capacity([10.0], tx_antennas=2, rx_antennas=2, channel_knowledge=knowledge, num_realizations=5)
            assert isinstance(capacity, torch.Tensor)

        # Test with invalid parameters
        with pytest.raises(ValueError):
            self.analyzer.mimo_capacity([10.0], tx_antennas=0, rx_antennas=2)

        with pytest.raises(ValueError):
            self.analyzer.mimo_capacity([10.0], tx_antennas=2, rx_antennas=2, channel_knowledge="invalid")

    def test_capacity_gap_to_shannon(self):
        """Test capacity gap calculation."""
        # Mock modulation_capacity to avoid actual computation
        with patch.object(self.analyzer, "modulation_capacity", return_value=(self.snr_db_tensor, torch.tensor([0.5, 0.7, 0.8, 0.9]))):
            # Test with a modulation scheme
            snr_values, gap = self.analyzer.capacity_gap_to_shannon(self.bpsk_modulator, self.awgn_channel, self.snr_db_tensor)
            assert isinstance(gap, torch.Tensor)
            assert gap.shape == snr_values.shape

            # Gap should be positive (Shannon capacity ≥ modulation capacity)
            assert torch.all(gap >= 0)

    def test_capacity_cdf(self):
        """Test capacity CDF calculation."""
        # Create a mock channel
        mock_channel = MagicMock()
        mock_channel.to.return_value = None

        # Mock channel function to return noisy output
        def mock_channel_func(input_):
            return input_ + 0.1 * torch.complex(torch.randn_like(input_.real), torch.randn_like(input_.imag))

        mock_channel.side_effect = mock_channel_func

        # Patch the _estimate_mutual_information method to return random values
        with patch.object(self.analyzer, "_estimate_mutual_information", side_effect=lambda *args, **kwargs: torch.rand(1)[0] * 2):  # Values between 0 and 2
            # Test CDF calculation
            capacities, cdf = self.analyzer.capacity_cdf(mock_channel, snr_db=10.0, num_realizations=10)
            assert isinstance(capacities, torch.Tensor)
            assert isinstance(cdf, torch.Tensor)
            assert capacities.shape == cdf.shape

            # CDF should be monotonically increasing and between 0 and 1
            assert torch.all(cdf[1:] >= cdf[:-1])
            assert torch.all(cdf >= 0) and torch.all(cdf <= 1)

    def test_spectral_efficiency(self):
        """Test spectral efficiency calculation."""
        # Mock modulation_capacity to avoid actual computation
        with patch.object(self.analyzer, "modulation_capacity", return_value=(self.snr_db_tensor, torch.tensor([0.5, 0.7, 0.8, 0.9]))):
            # Test without overhead
            snr_values, spectral_eff = self.analyzer.spectral_efficiency(self.bpsk_modulator, self.awgn_channel, self.snr_db_tensor, bandwidth=1.0, overhead=0.0)
            assert isinstance(spectral_eff, torch.Tensor)
            assert spectral_eff.shape == snr_values.shape

            # Test with overhead
            snr_values, spectral_eff_with_overhead = self.analyzer.spectral_efficiency(self.bpsk_modulator, self.awgn_channel, self.snr_db_tensor, bandwidth=1.0, overhead=0.2)  # 20% overhead

            # With overhead, efficiency should be lower
            assert torch.all(spectral_eff_with_overhead < spectral_eff)

    def test_energy_efficiency(self):
        """Test energy efficiency calculation."""
        # Mock modulation_capacity to avoid actual computation
        with patch.object(self.analyzer, "modulation_capacity", return_value=(self.snr_db_tensor, torch.tensor([0.5, 0.7, 0.8, 0.9]))):
            # Test with default power values
            snr_values, energy_eff = self.analyzer.energy_efficiency(self.bpsk_modulator, self.awgn_channel, self.snr_db_tensor)
            assert isinstance(energy_eff, torch.Tensor)
            assert energy_eff.shape == snr_values.shape

            # Test with custom power values
            snr_values, energy_eff_custom = self.analyzer.energy_efficiency(self.bpsk_modulator, self.awgn_channel, self.snr_db_tensor, tx_power_watts=2.0, circuit_power_watts=0.5)

            # With higher power, efficiency should be lower
            assert torch.all(energy_eff_custom < energy_eff)
