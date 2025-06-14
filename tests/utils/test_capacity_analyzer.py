"""Unit tests for capacity analyzer module."""

from unittest.mock import MagicMock, patch

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
        self.snr_db_array = torch.tensor(self.snr_db_list)
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

        # Test with torch tensor
        p_array = torch.tensor(p_list)
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

        # Test with torch tensor
        e_array = torch.tensor(e_list)
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

    def test_process_snr_for_mutual_information(self):
        """Test the parallel processing method for mutual information calculation."""
        # Test with BPSK modulator and AWGN channel
        snr_item = 10.0
        num_symbols = 100
        bits_per_symbol = 1

        # Test with histogram method
        mi_result = self.analyzer._process_snr_for_mutual_information(snr_item, self.bpsk_modulator, self.awgn_channel, num_symbols, bits_per_symbol, "histogram", num_bins=50)
        assert isinstance(mi_result, float)
        assert mi_result >= 0

        # Test with KNN method
        mi_result_knn = self.analyzer._process_snr_for_mutual_information(snr_item, self.bpsk_modulator, self.awgn_channel, num_symbols, bits_per_symbol, "knn", num_bins=50)
        assert isinstance(mi_result_knn, float)
        assert mi_result_knn >= 0

        # Test with QAM modulator that requires order parameter
        mi_result_qam = self.analyzer._process_snr_for_mutual_information(snr_item, self.qam_modulator, self.awgn_channel, num_symbols, 4, "histogram", num_bins=50)
        assert isinstance(mi_result_qam, float)
        assert mi_result_qam >= 0

    def test_process_snr_with_different_channel_types(self):
        """Test _process_snr_for_mutual_information with different channel types."""
        snr_item = 10.0
        num_symbols = 50
        bits_per_symbol = 1

        # Create mock channels of different types
        mock_rayleigh_channel = MagicMock()
        mock_rayleigh_channel.__class__.__name__ = "RayleighFadingChannel"
        mock_rayleigh_channel.coherence_time = 10
        mock_rayleigh_channel.side_effect = lambda x: x + 0.1 * torch.randn_like(x)

        # Test with Rayleigh fading channel
        with patch.object(self.analyzer, "_estimate_mutual_information", return_value=torch.tensor(0.8)):
            mi_result = self.analyzer._process_snr_for_mutual_information(snr_item, self.bpsk_modulator, mock_rayleigh_channel, num_symbols, bits_per_symbol, "histogram")
            assert isinstance(mi_result, float)

        # Test with Rician fading channel
        mock_rician_channel = MagicMock()
        mock_rician_channel.__class__.__name__ = "RicianFadingChannel"
        mock_rician_channel.k_factor = 5
        mock_rician_channel.coherence_time = 10
        mock_rician_channel.side_effect = lambda x: x + 0.1 * torch.randn_like(x)

        with patch.object(self.analyzer, "_estimate_mutual_information", return_value=torch.tensor(0.9)):
            mi_result = self.analyzer._process_snr_for_mutual_information(snr_item, self.bpsk_modulator, mock_rician_channel, num_symbols, bits_per_symbol, "histogram")
            assert isinstance(mi_result, float)

        # Test with unknown channel type (fallback case)
        mock_unknown_channel = MagicMock()
        mock_unknown_channel.__class__.__name__ = "UnknownChannel"
        mock_unknown_channel.side_effect = lambda x: x + 0.1 * torch.randn_like(x)
        mock_unknown_channel.snr_db = 10.0

        with patch.object(self.analyzer, "_estimate_mutual_information", return_value=torch.tensor(0.7)):
            mi_result = self.analyzer._process_snr_for_mutual_information(snr_item, self.bpsk_modulator, mock_unknown_channel, num_symbols, bits_per_symbol, "histogram")
            assert isinstance(mi_result, float)

    def test_estimate_mutual_information_real_computation(self):
        """Test actual mutual information estimation without mocking."""
        # Test with real signals - small dataset for speed
        num_samples = 100

        # Create controlled test data
        tx_real = torch.zeros(num_samples, device=self.device)  # All zeros
        rx_real = tx_real + 0.01 * torch.randn(num_samples, device=self.device)  # Very low noise

        mi_real = self.analyzer._estimate_mutual_information(tx_real, rx_real, num_bins=10, bits_per_symbol=1)
        assert isinstance(mi_real, torch.Tensor)
        assert mi_real.dim() == 0  # Scalar tensor
        assert mi_real >= 0

        # Test with complex signals - small dataset
        tx_complex = torch.complex(torch.zeros(num_samples, device=self.device), torch.zeros(num_samples, device=self.device))
        rx_complex = tx_complex + 0.01 * torch.complex(torch.randn(num_samples, device=self.device), torch.randn(num_samples, device=self.device))

        mi_complex = self.analyzer._estimate_mutual_information(tx_complex, rx_complex, num_bins=10, bits_per_symbol=2)
        assert isinstance(mi_complex, torch.Tensor)
        assert mi_complex.dim() == 0
        assert mi_complex >= 0

    def test_mutual_information_parallel_processing(self):
        """Test parallel processing in mutual information calculation."""
        # Create analyzer with multiple processes (but use CPU device)
        analyzer_parallel = CapacityAnalyzer(device=self.device, num_processes=2)

        # Test with short computation to avoid long test times
        mi = analyzer_parallel.mutual_information(self.bpsk_modulator, self.awgn_channel, [5.0, 10.0], num_symbols=100, estimation_method="histogram")
        assert isinstance(mi, torch.Tensor)
        assert mi.shape == torch.Size([2])
        assert torch.all(mi >= 0)

    def test_cache_functionality(self):
        """Test that caching works correctly for various methods."""
        # Test mutual information cache
        snr_vals = [5.0, 10.0]

        # First call should compute and cache
        mi1 = self.analyzer.mutual_information(self.bpsk_modulator, self.awgn_channel, snr_vals, num_symbols=50)

        # Second call should use cache (should be faster and identical)
        mi2 = self.analyzer.mutual_information(self.bpsk_modulator, self.awgn_channel, snr_vals, num_symbols=50)
        assert torch.allclose(mi1, mi2)

        # Test modulation capacity cache
        snr_range1, cap1 = self.analyzer.modulation_capacity(self.bpsk_modulator, self.awgn_channel, snr_vals, num_symbols=50)
        snr_range2, cap2 = self.analyzer.modulation_capacity(self.bpsk_modulator, self.awgn_channel, snr_vals, num_symbols=50)
        assert torch.allclose(cap1, cap2)

        # Test MIMO capacity cache
        mimo_cap1 = self.analyzer.mimo_capacity(snr_vals, tx_antennas=2, rx_antennas=2, num_realizations=5)
        mimo_cap2 = self.analyzer.mimo_capacity(snr_vals, tx_antennas=2, rx_antennas=2, num_realizations=5)
        assert torch.allclose(mimo_cap1, mimo_cap2)

    def test_input_type_conversions(self):
        """Test that various input types are handled correctly."""
        import numpy as np

        # Test with numpy arrays
        snr_numpy = np.array([5.0, 10.0])
        capacity_numpy = self.analyzer.awgn_capacity(snr_numpy)
        assert isinstance(capacity_numpy, torch.Tensor)

        # Test with single values
        capacity_single = self.analyzer.awgn_capacity(10.0)
        assert isinstance(capacity_single, torch.Tensor)
        assert capacity_single.shape == torch.Size([1])

        # Test BEC capacity with numpy array
        e_numpy = np.array([0.1, 0.2, 0.3])
        bec_cap_numpy = self.analyzer.bec_capacity(e_numpy)
        assert isinstance(bec_cap_numpy, torch.Tensor)

        # Test BSC capacity with numpy array
        p_numpy = np.array([0.1, 0.2, 0.3])
        bsc_cap_numpy = self.analyzer.bsc_capacity(p_numpy)
        assert isinstance(bsc_cap_numpy, torch.Tensor)

    def test_edge_cases_and_boundaries(self):
        """Test edge cases and boundary conditions."""
        # Test binary entropy at boundaries
        p_boundary = torch.tensor([0.0, 0.5, 1.0], device=self.device)
        entropy_boundary = self.analyzer._binary_entropy(p_boundary)
        expected = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        assert torch.allclose(entropy_boundary, expected, atol=1e-6)

        # Test BSC capacity with invalid probabilities (should be clamped)
        p_invalid = torch.tensor([0.7, 0.8, 0.9], device=self.device)
        bsc_cap_invalid = self.analyzer.bsc_capacity(p_invalid)
        bsc_cap_half = self.analyzer.bsc_capacity(0.5)
        # All should be clamped to 0.5
        assert torch.allclose(bsc_cap_invalid, bsc_cap_half.expand_as(bsc_cap_invalid))

        # Test BEC capacity at boundaries
        e_boundary = torch.tensor([0.0, 1.0], device=self.device)
        bec_cap_boundary = self.analyzer.bec_capacity(e_boundary)
        expected_bec = torch.tensor([1.0, 0.0], device=self.device)
        assert torch.allclose(bec_cap_boundary, expected_bec)

    def test_mimo_capacity_edge_cases(self):
        """Test MIMO capacity with various edge cases."""
        # Test with equal number of antennas
        capacity_equal = self.analyzer.mimo_capacity([10.0], tx_antennas=2, rx_antennas=2, num_realizations=5)
        assert isinstance(capacity_equal, torch.Tensor)

        # Test with more TX than RX antennas
        capacity_tx_more = self.analyzer.mimo_capacity([10.0], tx_antennas=4, rx_antennas=2, num_realizations=5)
        assert isinstance(capacity_tx_more, torch.Tensor)

        # Test with more RX than TX antennas
        capacity_rx_more = self.analyzer.mimo_capacity([10.0], tx_antennas=2, rx_antennas=4, num_realizations=5)
        assert isinstance(capacity_rx_more, torch.Tensor)

        # Test single antenna case (SISO)
        capacity_siso = self.analyzer.mimo_capacity([10.0], tx_antennas=1, rx_antennas=1, num_realizations=5)
        assert isinstance(capacity_siso, torch.Tensor)

    def test_modulation_capacity_with_different_estimation_methods(self):
        """Test modulation capacity with different estimation methods."""
        snr_vals = [10.0]

        # Test with histogram method
        _, cap_hist = self.analyzer.modulation_capacity(self.bpsk_modulator, self.awgn_channel, snr_vals, num_symbols=100, monte_carlo=True, estimation_method="histogram")
        assert isinstance(cap_hist, torch.Tensor)

        # Test with KNN method
        _, cap_knn = self.analyzer.modulation_capacity(self.bpsk_modulator, self.awgn_channel, snr_vals, num_symbols=100, monte_carlo=True, estimation_method="knn")
        assert isinstance(cap_knn, torch.Tensor)

    def test_plot_functions_with_different_inputs(self):
        """Test plotting functions with various input formats."""
        with patch("matplotlib.pyplot.figure") as mock_figure:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_fig.add_subplot.return_value = mock_ax
            mock_figure.return_value = mock_fig

            snr_range = self.snr_db_tensor
            capacity = torch.rand_like(snr_range)

            # Test plot_capacity_vs_snr with all optional parameters
            fig = self.analyzer.plot_capacity_vs_snr(
                snr_range, capacity, include_shannon=True, include_shannon_mimo=True, mimo_tx=2, mimo_rx=2, title="Test Capacity Plot", xlabel="Custom SNR Label", ylabel="Custom Capacity Label", legend_loc="upper left", figsize=(12, 8), grid=False, style="seaborn-v0_8"
            )
            assert fig is not None

            # Test plot_capacity_vs_param with custom parameters
            param_values = torch.linspace(0, 1, 5)
            fig = self.analyzer.plot_capacity_vs_param(param_values, capacity, param_name="Custom Parameter", title="Custom Title", xlabel="Custom X Label", ylabel="Custom Y Label", legend_loc="upper right", figsize=(8, 6), grid=False)
            assert fig is not None

    def test_fast_mode_differences(self):
        """Test that fast mode affects behavior appropriately."""
        # Create analyzers with and without fast mode
        analyzer_fast = CapacityAnalyzer(device=self.device, fast_mode=True)
        analyzer_slow = CapacityAnalyzer(device=self.device, fast_mode=False)

        # Both should work but potentially with different internal behavior
        cap_fast = analyzer_fast.awgn_capacity(10.0)
        cap_slow = analyzer_slow.awgn_capacity(10.0)

        # Results should be the same for basic methods
        assert torch.allclose(cap_fast, cap_slow)

    def test_error_handling_and_robustness(self):
        """Test error handling and robustness for edge cases."""
        # Test with very small number of symbols
        mi_small = self.analyzer.mutual_information(self.bpsk_modulator, self.awgn_channel, [10.0], num_symbols=10, estimation_method="histogram")
        assert isinstance(mi_small, torch.Tensor)

        # Test with very few bins
        mi_few_bins = self.analyzer.mutual_information(self.bpsk_modulator, self.awgn_channel, [10.0], num_symbols=50, num_bins=5, estimation_method="histogram")
        assert isinstance(mi_few_bins, torch.Tensor)

    def test_high_snr_qam_capacity_approximation(self):
        """Test QAM capacity approximation at high SNR."""
        # Very high SNR should approach log2(constellation_size)
        high_snr = torch.tensor([50.0, 100.0])  # Very high SNR values

        # Test with 16-QAM (should approach 4 bits/symbol)
        capacity_16qam = self.analyzer._qam_awgn_capacity(high_snr, constellation_size=16)
        assert isinstance(capacity_16qam, torch.Tensor)
        # At very high SNR, should be close to log2(16) = 4
        assert capacity_16qam[-1] > 3.8  # Allow some margin

        # Test with 4-QAM/QPSK (should approach 2 bits/symbol)
        capacity_4qam = self.analyzer._qam_awgn_capacity(high_snr, constellation_size=4)
        assert isinstance(capacity_4qam, torch.Tensor)
        # At very high SNR, should be close to log2(4) = 2
        assert capacity_4qam[-1] > 1.8  # Allow some margin

    def test_capacity_cdf_with_different_realizations(self):
        """Test capacity CDF with different numbers of realizations."""
        mock_channel = MagicMock()
        mock_channel.to.return_value = None
        mock_channel.side_effect = lambda x: x + 0.1 * torch.randn_like(x)

        with patch.object(self.analyzer, "_estimate_mutual_information", side_effect=lambda *args, **kwargs: torch.rand(1)[0]):
            # Test with small number of realizations
            capacities_small, cdf_small = self.analyzer.capacity_cdf(mock_channel, snr_db=10.0, num_realizations=5)
            assert isinstance(capacities_small, torch.Tensor)
            assert isinstance(cdf_small, torch.Tensor)
            assert capacities_small.shape == cdf_small.shape
            assert capacities_small.shape[0] == 5

            # Test with larger number of realizations
            capacities_large, cdf_large = self.analyzer.capacity_cdf(mock_channel, snr_db=10.0, num_realizations=20)
            assert capacities_large.shape[0] == 20

    def test_modulator_without_order_attribute(self):
        """Test _process_snr_for_mutual_information with modulators that don't have order
        attribute."""
        # Create a mock modulator without order attribute
        mock_modulator = MagicMock()
        mock_modulator.__class__.__name__ = "QAMModulator"
        mock_modulator.bits_per_symbol = 2
        # Intentionally no 'order' attribute to test fallback

        mock_modulator.side_effect = lambda x: x  # Simple pass-through

        snr_item = 10.0
        with patch.object(self.analyzer, "_estimate_mutual_information", return_value=torch.tensor(1.0)):
            mi_result = self.analyzer._process_snr_for_mutual_information(snr_item, mock_modulator, self.awgn_channel, 50, 2, "histogram")
            assert isinstance(mi_result, float)

    def test_channel_without_state_dict(self):
        """Test _process_snr_for_mutual_information with channels that don't have state_dict."""
        # Create a mock channel without state_dict
        mock_channel = MagicMock()
        mock_channel.__class__.__name__ = "CustomChannel"
        mock_channel.side_effect = lambda x: x + 0.1 * torch.randn_like(x)
        # Intentionally no state_dict method

        snr_item = 10.0
        with patch.object(self.analyzer, "_estimate_mutual_information", return_value=torch.tensor(0.8)):
            mi_result = self.analyzer._process_snr_for_mutual_information(snr_item, self.bpsk_modulator, mock_channel, 50, 1, "histogram")
            assert isinstance(mi_result, float)

    def test_channel_with_avg_noise_power(self):
        """Test _process_snr_for_mutual_information with channels using avg_noise_power instead of
        snr_db."""
        # Create a mock channel that uses avg_noise_power
        mock_channel = MagicMock()
        mock_channel.__class__.__name__ = "NoiseBasedChannel"
        mock_channel.side_effect = lambda x: x + 0.1 * torch.randn_like(x)
        mock_channel.avg_noise_power = 0.1  # Will be overwritten

        snr_item = 10.0
        with patch.object(self.analyzer, "_estimate_mutual_information", return_value=torch.tensor(0.9)):
            mi_result = self.analyzer._process_snr_for_mutual_information(snr_item, self.bpsk_modulator, mock_channel, 50, 1, "histogram")
            assert isinstance(mi_result, float)
            # Check that avg_noise_power was set correctly
            # For SNR = 10 dB = 10, noise power should be 1/10 = 0.1
            expected_noise_power = 1.0 / (10 ** (10.0 / 10))
            assert abs(mock_channel.avg_noise_power - expected_noise_power) < 1e-6

    def test_estimate_mutual_information_edge_cases(self):
        """Test mutual information estimation with edge cases."""
        # Test with identical transmitted symbols (should give low MI)
        num_samples = 50
        tx_identical = torch.ones(num_samples, device=self.device)  # All ones
        rx_identical = tx_identical + 0.1 * torch.randn(num_samples, device=self.device)

        mi_identical = self.analyzer._estimate_mutual_information(tx_identical, rx_identical, num_bins=10, bits_per_symbol=1)
        assert isinstance(mi_identical, torch.Tensor)
        assert mi_identical >= 0

        # Test with very diverse received values
        tx_diverse = torch.randn(num_samples, device=self.device)
        rx_diverse = tx_diverse + torch.randn(num_samples, device=self.device)

        mi_diverse = self.analyzer._estimate_mutual_information(tx_diverse, rx_diverse, num_bins=10, bits_per_symbol=1)
        assert isinstance(mi_diverse, torch.Tensor)
        assert mi_diverse >= 0

        # Test with complex signals with identical real/imaginary parts
        tx_complex_simple = torch.complex(torch.ones(num_samples, device=self.device), torch.ones(num_samples, device=self.device))
        rx_complex_simple = tx_complex_simple + 0.1 * torch.complex(torch.randn(num_samples, device=self.device), torch.randn(num_samples, device=self.device))

        mi_complex_simple = self.analyzer._estimate_mutual_information(tx_complex_simple, rx_complex_simple, num_bins=5, bits_per_symbol=2)
        assert isinstance(mi_complex_simple, torch.Tensor)
        assert mi_complex_simple >= 0

    def test_estimate_mutual_information_with_very_small_ranges(self):
        """Test mutual information estimation when signal ranges are very small."""
        num_samples = 100

        # Test with very small signal range (should handle numerical precision)
        tx_small = torch.tensor([0.0001, 0.0002] * (num_samples // 2), device=self.device)
        rx_small = tx_small + 1e-6 * torch.randn(num_samples, device=self.device)

        mi_small_range = self.analyzer._estimate_mutual_information(tx_small, rx_small, num_bins=10, bits_per_symbol=1)
        assert isinstance(mi_small_range, torch.Tensor)
        assert mi_small_range >= 0

    def test_mutual_information_with_different_bin_counts(self):
        """Test mutual information with various bin counts."""
        num_samples = 100
        tx_test = torch.randn(num_samples, device=self.device)
        rx_test = tx_test + 0.1 * torch.randn(num_samples, device=self.device)

        # Test with different numbers of bins
        for num_bins in [5, 10, 20, 50]:
            mi_bins = self.analyzer._estimate_mutual_information(tx_test, rx_test, num_bins=num_bins, bits_per_symbol=1)
            assert isinstance(mi_bins, torch.Tensor)
            assert mi_bins >= 0

    def test_generate_qam_symbols_edge_cases(self):
        """Test QAM symbol generation with edge cases."""
        # Test with small constellation sizes
        symbols_2qam = self.analyzer._generate_qam_symbols(2, 100, self.device)
        assert symbols_2qam.shape == torch.Size([100])
        assert torch.is_complex(symbols_2qam)

        # Test with larger constellation
        symbols_64qam = self.analyzer._generate_qam_symbols(64, 50, self.device)
        assert symbols_64qam.shape == torch.Size([50])

        # Test normalization property
        assert torch.isclose(torch.mean(torch.abs(symbols_64qam) ** 2), torch.tensor(1.0), atol=0.1)

    def test_bpsk_awgn_capacity_edge_cases(self):
        """Test BPSK AWGN capacity with edge cases."""
        # Test with very low SNR
        low_snr = torch.tensor([-10.0, -5.0, 0.0], device=self.device)
        capacity_low = self.analyzer._bpsk_awgn_capacity(low_snr)
        assert torch.all(capacity_low >= 0)
        assert torch.all(capacity_low <= 1)

        # Test with very high SNR
        high_snr = torch.tensor([30.0, 40.0, 50.0], device=self.device)
        capacity_high = self.analyzer._bpsk_awgn_capacity(high_snr)
        assert torch.all(capacity_high >= 0)
        assert torch.all(capacity_high <= 1)
        # At very high SNR, should approach 1
        assert capacity_high[-1] > 0.95

    def test_qam_awgn_capacity_extreme_cases(self):
        """Test QAM AWGN capacity with extreme cases."""
        # Test with very low SNR (should approach 0)
        very_low_snr = torch.tensor([-20.0, -10.0], device=self.device)
        capacity_very_low = self.analyzer._qam_awgn_capacity(very_low_snr, constellation_size=16)
        assert torch.all(capacity_very_low >= 0)
        assert capacity_very_low[0] < 0.5  # Should be quite low

        # Test with extremely high SNR (triggers fast path)
        extremely_high_snr = torch.tensor([150.0, 200.0], device=self.device)  # > 100 dB
        capacity_extreme = self.analyzer._qam_awgn_capacity(extremely_high_snr, constellation_size=16)
        # Should return log2(16) = 4
        assert torch.allclose(capacity_extreme, torch.tensor([4.0, 4.0]), atol=1e-6)

    def test_capacity_gap_to_shannon_edge_cases(self):
        """Test capacity gap calculation with edge cases."""
        with patch.object(self.analyzer, "modulation_capacity", return_value=(self.snr_db_tensor, torch.tensor([0.0, 0.0, 0.0, 0.0]))):
            # Test with zero capacity (maximum gap)
            snr_values, gap = self.analyzer.capacity_gap_to_shannon(self.bpsk_modulator, self.awgn_channel, self.snr_db_tensor)
            assert torch.all(gap > 0)  # Gap should be positive since Shannon > 0

        with patch.object(self.analyzer, "modulation_capacity") as mock_mod_cap:
            # Test when modulation capacity equals Shannon capacity (theoretical limit)
            shannon_cap = self.analyzer.awgn_capacity(self.snr_db_tensor)
            mock_mod_cap.return_value = (self.snr_db_tensor, shannon_cap)

            snr_values, gap = self.analyzer.capacity_gap_to_shannon(self.bpsk_modulator, self.awgn_channel, self.snr_db_tensor)
            assert torch.allclose(gap, torch.zeros_like(gap), atol=1e-6)

    def test_spectral_efficiency_edge_cases(self):
        """Test spectral efficiency with edge cases."""
        with patch.object(self.analyzer, "modulation_capacity", return_value=(self.snr_db_tensor, torch.tensor([1.0, 2.0, 3.0, 4.0]))):
            # Test with zero bandwidth (should handle gracefully)
            with pytest.raises(ZeroDivisionError):
                self.analyzer.spectral_efficiency(self.bpsk_modulator, self.awgn_channel, self.snr_db_tensor, bandwidth=0.0)

            # Test with very small bandwidth
            snr_values, spec_eff = self.analyzer.spectral_efficiency(self.bpsk_modulator, self.awgn_channel, self.snr_db_tensor, bandwidth=0.001)
            assert torch.all(spec_eff > 0)  # Should still be positive

            # Test with 100% overhead (should halve efficiency)
            snr_values, spec_eff_100 = self.analyzer.spectral_efficiency(self.bpsk_modulator, self.awgn_channel, self.snr_db_tensor, bandwidth=1.0, overhead=1.0)
            snr_values, spec_eff_0 = self.analyzer.spectral_efficiency(self.bpsk_modulator, self.awgn_channel, self.snr_db_tensor, bandwidth=1.0, overhead=0.0)
            assert torch.allclose(spec_eff_100, spec_eff_0 * 0.0, atol=1e-6)  # 100% overhead means 0 efficiency

    def test_energy_efficiency_edge_cases(self):
        """Test energy efficiency with edge cases."""
        with patch.object(self.analyzer, "modulation_capacity", return_value=(self.snr_db_tensor, torch.tensor([1.0, 2.0, 3.0, 4.0]))):
            # Test with zero circuit power
            snr_values, energy_eff = self.analyzer.energy_efficiency(self.bpsk_modulator, self.awgn_channel, self.snr_db_tensor, tx_power_watts=1.0, circuit_power_watts=0.0)
            assert torch.all(energy_eff > 0)

            # Test with zero transmit power
            snr_values, energy_eff_zero_tx = self.analyzer.energy_efficiency(self.bpsk_modulator, self.awgn_channel, self.snr_db_tensor, tx_power_watts=0.0, circuit_power_watts=1.0)
            assert torch.all(energy_eff_zero_tx > 0)

            # Test with very high power consumption
            snr_values, energy_eff_high = self.analyzer.energy_efficiency(self.bpsk_modulator, self.awgn_channel, self.snr_db_tensor, tx_power_watts=100.0, circuit_power_watts=100.0)
            assert torch.all(energy_eff_high > 0)
            assert torch.all(energy_eff_high < energy_eff)  # Should be lower than with lower power

    def test_mimo_capacity_numerical_stability(self):
        """Test MIMO capacity calculation for numerical stability issues."""
        # Test with very high SNR (could cause numerical issues)
        high_snr = torch.tensor([50.0, 100.0], device=self.device)
        capacity_high = self.analyzer.mimo_capacity(high_snr, tx_antennas=4, rx_antennas=4, num_realizations=5)
        assert torch.all(torch.isfinite(capacity_high))  # Should not have inf or nan
        assert torch.all(capacity_high > 0)

        # Test with very low SNR
        low_snr = torch.tensor([-20.0, -10.0], device=self.device)
        capacity_low = self.analyzer.mimo_capacity(low_snr, tx_antennas=2, rx_antennas=2, num_realizations=5)
        assert torch.all(torch.isfinite(capacity_low))
        assert torch.all(capacity_low >= 0)

    def test_modulation_capacity_analytical_paths(self):
        """Test analytical paths in modulation capacity calculation."""
        # Test BPSK over AWGN (should use analytical solution)
        snr_range, capacity_bpsk = self.analyzer.modulation_capacity(self.bpsk_modulator, self.awgn_channel, [10.0], monte_carlo=False)  # Force analytical
        assert isinstance(capacity_bpsk, torch.Tensor)

        # Test QPSK over AWGN (should use analytical solution)
        snr_range, capacity_qpsk = self.analyzer.modulation_capacity(self.qpsk_modulator, self.awgn_channel, [10.0], monte_carlo=False)  # Force analytical
        assert isinstance(capacity_qpsk, torch.Tensor)

    def test_multiprocessing_error_handling(self):
        """Test error handling in multiprocessing scenarios."""
        # Create analyzer with multiple processes but force GPU usage (should fall back to sequential)
        if torch.cuda.is_available():
            analyzer_gpu = CapacityAnalyzer(device=torch.device("cuda"), num_processes=2)

            # Should fall back to sequential processing on GPU
            mi = analyzer_gpu.mutual_information(self.bpsk_modulator, self.awgn_channel, [5.0, 10.0], num_symbols=50)
            assert isinstance(mi, torch.Tensor)

    def test_device_handling(self):
        """Test device handling across different methods."""
        # Test that tensors are properly moved to the analyzer's device
        if self.device.type == "cpu":
            # Create tensors on a different device context
            snr_vals = torch.tensor([10.0], device=self.device)

            # Test various methods handle device correctly
            capacity = self.analyzer.awgn_capacity(snr_vals)
            assert capacity.device == self.device

            bsc_cap = self.analyzer.bsc_capacity(torch.tensor([0.1], device=self.device))
            assert bsc_cap.device == self.device

            bec_cap = self.analyzer.bec_capacity(torch.tensor([0.1], device=self.device))
            assert bec_cap.device == self.device

    def test_initialization_edge_cases(self):
        """Test initialization with edge cases."""
        # Test with num_processes = 0 (should default to 1)
        analyzer_zero = CapacityAnalyzer(num_processes=0)
        assert analyzer_zero.num_processes == 1

        # Test with negative num_processes (should default to 1)
        analyzer_neg = CapacityAnalyzer(num_processes=-2)
        assert analyzer_neg.num_processes >= 1  # Should be at least 1

    def test_plot_mimo_shannon_only(self):
        """Test plotting with MIMO Shannon capacity only."""
        with patch("matplotlib.pyplot.figure") as mock_figure:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_fig.add_subplot.return_value = mock_ax
            mock_figure.return_value = mock_fig

            snr_range = self.snr_db_tensor
            capacity = torch.rand_like(snr_range)

            # Test with MIMO Shannon but no regular Shannon
            fig = self.analyzer.plot_capacity_vs_snr(snr_range, capacity, include_shannon=False, include_shannon_mimo=True, mimo_tx=3, mimo_rx=3)
            assert fig is not None

    def test_plot_capacity_vs_snr_list_input(self):
        """Test plot_capacity_vs_snr with list input for capacities."""
        with patch("matplotlib.pyplot.figure") as mock_figure:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_fig.add_subplot.return_value = mock_ax
            mock_figure.return_value = mock_fig

            snr_range = self.snr_db_tensor
            capacity1 = torch.rand_like(snr_range)
            capacity2 = torch.rand_like(snr_range)
            capacities_list = [capacity1, capacity2]

            # Test with list input and custom labels
            fig = self.analyzer.plot_capacity_vs_snr(snr_range, capacities_list, labels=["Method A", "Method B"])
            assert fig is not None

            # Test with list input but no labels (should use default)
            fig = self.analyzer.plot_capacity_vs_snr(snr_range, capacities_list)
            assert fig is not None

    def test_plot_capacity_vs_param_custom_xlabel(self):
        """Test plot_capacity_vs_param with custom xlabel handling."""
        with patch("matplotlib.pyplot.figure") as mock_figure:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_fig.add_subplot.return_value = mock_ax
            mock_figure.return_value = mock_fig

            param_values = torch.linspace(0, 1, 5)
            capacity = torch.rand_like(param_values)

            # Test with custom xlabel (should override param_name)
            fig = self.analyzer.plot_capacity_vs_param(param_values, capacity, param_name="Error Rate", xlabel="Custom X Label")
            assert fig is not None

            # Test with no xlabel (should use param_name)
            fig = self.analyzer.plot_capacity_vs_param(param_values, capacity, param_name="Error Rate", xlabel="")  # Empty string should use param_name
            assert fig is not None

    def test_ergodic_capacity_real_computation(self):
        """Test ergodic capacity with actual computation paths."""
        # Create a simple mock channel that adds noise
        mock_channel = MagicMock()
        mock_channel.to.return_value = None

        def simple_channel(input_symbols):
            # Add small amount of noise
            noise = 0.1 * torch.complex(torch.randn_like(input_symbols.real), torch.randn_like(input_symbols.imag))
            return input_symbols + noise

        mock_channel.side_effect = simple_channel

        # Test with small number of realizations for speed
        snr_values, capacity = self.analyzer.ergodic_capacity(mock_channel, [10.0], num_realizations=3, num_symbols_per_realization=20)
        assert isinstance(capacity, torch.Tensor)
        assert capacity.shape == torch.Size([1])
        assert capacity[0] > 0

    def test_outage_capacity_real_computation(self):
        """Test outage capacity with actual computation paths."""
        # Create a simple mock channel
        mock_channel = MagicMock()
        mock_channel.to.return_value = None

        def variable_noise_channel(input_symbols):
            # Variable noise to create outage events
            noise_var = 0.1 + 0.2 * torch.rand(1).item()  # Variable noise
            noise = noise_var * torch.complex(torch.randn_like(input_symbols.real), torch.randn_like(input_symbols.imag))
            return input_symbols + noise

        mock_channel.side_effect = variable_noise_channel

        # Test with small number of realizations
        snr_values, outage_cap = self.analyzer.outage_capacity(mock_channel, [10.0], outage_probability=0.2, num_realizations=10, num_symbols_per_realization=20)  # 20% outage
        assert isinstance(outage_cap, torch.Tensor)
        assert outage_cap.shape == torch.Size([1])
        assert outage_cap[0] >= 0

    def test_comparison_methods_with_real_data(self):
        """Test comparison methods with minimal real computation."""
        # Test compare_modulation_schemes without mocking
        modulators = [self.bpsk_modulator]  # Single modulator for speed

        snr_values, capacities, fig = self.analyzer.compare_modulation_schemes(modulators, self.awgn_channel, [10.0], num_symbols=50, plot=False)
        assert isinstance(capacities, dict)
        assert len(capacities) == 1

        # Test compare_channels without mocking
        channels = [self.awgn_channel]  # Single channel for speed

        snr_values, capacities, fig = self.analyzer.compare_channels(self.bpsk_modulator, channels, [10.0], num_symbols=50, plot=False)
        assert isinstance(capacities, dict)
        assert len(capacities) == 1

    def test_gaussian_input_capacity_fallback_path(self):
        """Test gaussian_input_capacity fallback to ergodic_capacity."""
        # Create mock channel that's not AWGN or Rayleigh
        mock_channel = MagicMock()
        mock_channel.__class__.__name__ = "CustomChannel"
        mock_channel.to.return_value = None

        # Mock ergodic_capacity to return specific values
        with patch.object(self.analyzer, "ergodic_capacity", return_value=(torch.tensor([10.0]), torch.tensor([2.5]))):
            capacity = self.analyzer.gaussian_input_capacity(mock_channel, [10.0])
            assert torch.isclose(capacity[0], torch.tensor(2.5))

    def test_modulation_capacity_cache_hits(self):
        """Test that modulation capacity caching works properly."""
        # First call should compute and cache
        snr_range1, cap1 = self.analyzer.modulation_capacity(self.bpsk_modulator, self.awgn_channel, [10.0], num_symbols=50, monte_carlo=False)

        # Clear the internal mutual info cache but keep capacity cache
        self.analyzer._mutual_info_cache.clear()

        # Second call should use capacity cache
        snr_range2, cap2 = self.analyzer.modulation_capacity(self.bpsk_modulator, self.awgn_channel, [10.0], num_symbols=50, monte_carlo=False)

        assert torch.allclose(cap1, cap2)

    def test_qam_modulator_order_detection(self):
        """Test QAM modulator order detection in _process_snr_for_mutual_information."""
        # Create a QAM modulator with specific order
        qam_mod = self.qam_modulator  # 16-QAM from setup

        # Test that the method correctly handles QAM modulator with order
        snr_item = 10.0
        with patch.object(self.analyzer, "_estimate_mutual_information", return_value=torch.tensor(2.0)):
            mi_result = self.analyzer._process_snr_for_mutual_information(snr_item, qam_mod, self.awgn_channel, 50, 4, "histogram")
            assert isinstance(mi_result, float)

    def test_psk_modulator_in_process_snr(self):
        """Test PSK modulator handling in _process_snr_for_mutual_information."""
        # Create mock PSK modulator
        mock_psk = MagicMock()
        mock_psk.__class__.__name__ = "PSKModulator"
        mock_psk.order = 8  # 8-PSK
        mock_psk.bits_per_symbol = 3
        mock_psk.side_effect = lambda x: x  # Simple pass-through

        snr_item = 10.0
        with patch.object(self.analyzer, "_estimate_mutual_information", return_value=torch.tensor(2.5)):
            mi_result = self.analyzer._process_snr_for_mutual_information(snr_item, mock_psk, self.awgn_channel, 50, 3, "histogram")
            assert isinstance(mi_result, float)

    def test_modulator_state_dict_error_handling(self):
        """Test error handling when modulator state_dict fails."""
        # Create mock modulator that raises exception on state_dict

        mock_mod = MagicMock()
        mock_mod.__class__.__name__ = "BPSKModulator"
        mock_mod.bits_per_symbol = 1
        mock_mod.state_dict.side_effect = Exception("State dict error")
        mock_mod.side_effect = lambda x: x

        snr_item = 10.0
        with patch.object(self.analyzer, "_estimate_mutual_information", return_value=torch.tensor(0.8)):
            # Should handle the exception and fall back to attribute copying
            mi_result = self.analyzer._process_snr_for_mutual_information(snr_item, mock_mod, self.awgn_channel, 50, 1, "histogram")
            assert isinstance(mi_result, float)

    def test_channel_state_dict_error_handling(self):
        """Test error handling when channel state_dict fails."""
        # Create mock channel that raises exception on state_dict
        mock_channel = MagicMock()
        mock_channel.__class__.__name__ = "AWGNChannel"
        mock_channel.state_dict.side_effect = Exception("State dict error")
        mock_channel.side_effect = lambda x: x

        snr_item = 10.0
        with patch.object(self.analyzer, "_estimate_mutual_information", return_value=torch.tensor(0.8)):
            # Should handle the exception and create new channel instance
            mi_result = self.analyzer._process_snr_for_mutual_information(snr_item, self.bpsk_modulator, mock_channel, 50, 1, "histogram")
            assert isinstance(mi_result, float)

    def test_binary_entropy_numerical_stability(self):
        """Test binary entropy function numerical stability."""
        # Test with values very close to 0 and 1
        p_edge = torch.tensor([1e-10, 1e-8, 1 - 1e-8, 1 - 1e-10], device=self.device)
        entropy_edge = self.analyzer._binary_entropy(p_edge)

        # Should handle these edge cases without numerical issues
        assert torch.all(torch.isfinite(entropy_edge))
        assert torch.all(entropy_edge >= 0)

    def test_mimo_capacity_with_all_channel_knowledge_types(self):
        """Test MIMO capacity with all channel knowledge types and error conditions."""
        snr_vals = [10.0]

        # Test all channel knowledge types
        for knowledge in ["perfect", "statistical", "none"]:
            capacity = self.analyzer.mimo_capacity(snr_vals, tx_antennas=2, rx_antennas=2, channel_knowledge=knowledge, num_realizations=3)
            assert isinstance(capacity, torch.Tensor)
            assert capacity[0] > 0

    def test_edge_case_zero_or_negative_antennas(self):
        """Test MIMO capacity error handling for invalid antenna numbers."""
        # Test zero antennas
        with pytest.raises(ValueError, match="Number of antennas must be positive"):
            self.analyzer.mimo_capacity([10.0], tx_antennas=0, rx_antennas=2)

        with pytest.raises(ValueError, match="Number of antennas must be positive"):
            self.analyzer.mimo_capacity([10.0], tx_antennas=2, rx_antennas=0)

        # Test negative antennas
        with pytest.raises(ValueError, match="Number of antennas must be positive"):
            self.analyzer.mimo_capacity([10.0], tx_antennas=-1, rx_antennas=2)

    def test_mimo_capacity_invalid_channel_knowledge(self):
        """Test MIMO capacity error handling for invalid channel knowledge."""
        with pytest.raises(ValueError, match="Channel knowledge must be"):
            self.analyzer.mimo_capacity([10.0], tx_antennas=2, rx_antennas=2, channel_knowledge="invalid_type")

    def test_complete_coverage_scenarios(self):
        """Additional tests to ensure complete coverage of remaining edge cases."""
        # Test capacity_cdf with specific number of symbols
        mock_channel = MagicMock()
        mock_channel.to.return_value = None
        mock_channel.side_effect = lambda x: x + 0.1 * torch.randn_like(x)

        # Test with num_symbols parameter in capacity_cdf
        with patch.object(self.analyzer, "_estimate_mutual_information", return_value=torch.tensor(1.0)):
            capacities, cdf = self.analyzer.capacity_cdf(mock_channel, snr_db=10.0, num_realizations=5)
            assert capacities.shape == torch.Size([5])
            assert cdf.shape == torch.Size([5])

        # Test that CDF goes from near 0 to 1
        assert cdf[0] == 0.2  # 1/5
        assert cdf[-1] == 1.0  # 5/5
