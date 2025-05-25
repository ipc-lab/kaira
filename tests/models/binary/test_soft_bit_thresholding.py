"""Tests for soft bit thresholding modules."""

import pytest
import torch

from kaira.models.binary.soft_bit_thresholding import (
    AdaptiveThresholder,
    DynamicThresholder,
    FixedThresholder,
    HysteresisThresholder,
    InputType,
    LLRThresholder,
    MinDistanceThresholder,
    OutputType,
    RepetitionSoftBitDecoder,
    SoftBitEnsembleThresholder,
    SoftBitThresholder,
    WeightedThresholder,
)


class TestInputType:
    """Test InputType enum."""

    def test_input_type_values(self):
        """Test that InputType has correct values."""
        assert InputType.PROBABILITY == "prob"
        assert InputType.LLR == "llr"
        assert InputType.SOFT == "soft"


class TestOutputType:
    """Test OutputType enum."""

    def test_output_type_values(self):
        """Test that OutputType has correct values."""
        assert OutputType.HARD == "hard"
        assert OutputType.SOFT == "soft"


class TestFixedThresholder:
    """Test FixedThresholder class."""

    def test_init_default(self):
        """Test FixedThresholder initialization with default parameters."""
        thresholder = FixedThresholder()
        assert thresholder.threshold == 0.5
        assert thresholder.input_type == InputType.PROBABILITY

    def test_init_custom(self):
        """Test FixedThresholder initialization with custom parameters."""
        thresholder = FixedThresholder(threshold=0.3, input_type=InputType.LLR)
        assert thresholder.threshold == 0.3
        assert thresholder.input_type == InputType.LLR

    def test_forward_probability(self):
        """Test forward pass with probability inputs."""
        thresholder = FixedThresholder(threshold=0.5)
        x = torch.tensor([0.2, 0.7, 0.4, 0.9])
        result = thresholder(x)
        expected = torch.tensor([0.0, 1.0, 0.0, 1.0])
        torch.testing.assert_close(result, expected)

    def test_forward_llr(self):
        """Test forward pass with LLR inputs."""
        thresholder = FixedThresholder(threshold=0.0, input_type=InputType.LLR)
        x = torch.tensor([-1.0, 1.0, -0.5, 2.0])
        result = thresholder(x)
        # FixedThresholder uses same logic for LLR as probability (x > threshold)
        # So negative values give 0.0, positive values give 1.0
        expected = torch.tensor([0.0, 1.0, 0.0, 1.0])
        torch.testing.assert_close(result, expected)

    def test_forward_batch(self):
        """Test forward pass with batch inputs."""
        thresholder = FixedThresholder()
        x = torch.tensor([[0.2, 0.7], [0.4, 0.9]])
        result = thresholder(x)
        expected = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        torch.testing.assert_close(result, expected)

    def test_unsupported_input_type(self):
        """Test error handling for unsupported input type."""
        thresholder = FixedThresholder()
        thresholder.input_type = "invalid"
        x = torch.tensor([0.5])
        with pytest.raises(ValueError, match="Unsupported input_type"):
            thresholder(x)


class TestAdaptiveThresholder:
    """Test AdaptiveThresholder class."""

    def test_init_default(self):
        """Test AdaptiveThresholder initialization with default parameters."""
        thresholder = AdaptiveThresholder()
        assert thresholder.method == "mean"
        assert thresholder.scale_factor == 1.0
        assert thresholder.input_type == InputType.PROBABILITY

    def test_init_custom(self):
        """Test AdaptiveThresholder initialization with custom parameters."""
        thresholder = AdaptiveThresholder(method="median", scale_factor=0.8)
        assert thresholder.method == "median"
        assert thresholder.scale_factor == 0.8

    def test_init_invalid_method(self):
        """Test error handling for invalid method."""
        with pytest.raises(ValueError, match="Method must be one of"):
            AdaptiveThresholder(method="invalid")

    def test_forward_mean_method(self):
        """Test forward pass with mean method."""
        thresholder = AdaptiveThresholder(method="mean")
        x = torch.tensor([0.1, 0.2, 0.8, 0.9])  # mean = 0.5
        result = thresholder(x)
        # Should threshold at mean (0.5)
        expected = torch.tensor([0.0, 0.0, 1.0, 1.0])
        torch.testing.assert_close(result, expected)

    def test_forward_median_method(self):
        """Test forward pass with median method."""
        thresholder = AdaptiveThresholder(method="median")
        x = torch.tensor([0.1, 0.4, 0.6, 0.9])  # median = 0.5
        result = thresholder(x)
        expected = torch.tensor([0.0, 0.0, 1.0, 1.0])
        torch.testing.assert_close(result, expected)

    def test_forward_otsu_method(self):
        """Test forward pass with Otsu method."""
        thresholder = AdaptiveThresholder(method="otsu")
        # Create bimodal distribution
        x = torch.cat([torch.full((50,), 0.2), torch.full((50,), 0.8)])
        result = thresholder(x)
        # Most values should be thresholded correctly
        assert result.sum() > 40  # Most high values should be 1
        assert result.sum() < 60  # Most low values should be 0

    def test_forward_llr_input(self):
        """Test forward pass with LLR input."""
        thresholder = AdaptiveThresholder(input_type=InputType.LLR)
        x = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        result = thresholder(x)
        assert result.shape == x.shape

    def test_otsu_threshold_function(self):
        """Test Otsu threshold calculation."""
        thresholder = AdaptiveThresholder(method="otsu")
        # Test with known bimodal distribution
        x = torch.cat([torch.full((100,), 0.2), torch.full((100,), 0.8)])
        threshold = thresholder._otsu_threshold(x)
        # Otsu threshold for this distribution should be closer to the smaller peak
        assert 0.15 < threshold < 0.25


class TestLLRThresholder:
    """Test LLRThresholder class."""

    def test_init_default(self):
        """Test LLRThresholder initialization with default parameters."""
        thresholder = LLRThresholder()
        assert thresholder.threshold == 0.0
        assert thresholder.confidence_scaling == 1.0
        assert thresholder.output_type == OutputType.HARD

    def test_init_custom(self):
        """Test LLRThresholder initialization with custom parameters."""
        thresholder = LLRThresholder(threshold=0.5, confidence_scaling=2.0, output_type=OutputType.SOFT)
        assert thresholder.threshold == 0.5
        assert thresholder.confidence_scaling == 2.0
        assert thresholder.output_type == OutputType.SOFT

    def test_forward_hard_output(self):
        """Test forward pass with hard output."""
        thresholder = LLRThresholder(threshold=0.0)
        x = torch.tensor([-1.0, 1.0, -0.5, 2.0])
        result = thresholder(x)
        expected = torch.tensor([1.0, 0.0, 1.0, 0.0])
        torch.testing.assert_close(result, expected)

    def test_forward_soft_output(self):
        """Test forward pass with soft output."""
        thresholder = LLRThresholder(output_type=OutputType.SOFT)
        x = torch.tensor([-1.0, 1.0, 0.0])
        result = thresholder(x)
        # Should convert to probabilities
        assert torch.all((result >= 0) & (result <= 1))
        assert result[0] > result[1]  # Negative LLR should have higher prob

    def test_confidence_scaling(self):
        """Test confidence scaling effect."""
        thresholder = LLRThresholder(confidence_scaling=2.0, output_type=OutputType.SOFT)
        x = torch.tensor([1.0])
        result = thresholder(x)

        thresholder_no_scaling = LLRThresholder(confidence_scaling=1.0, output_type=OutputType.SOFT)
        result_no_scaling = thresholder_no_scaling(x)

        # Scaling should change the output
        assert not torch.allclose(result, result_no_scaling)

    def test_unsupported_output_type(self):
        """Test error handling for unsupported output type."""
        thresholder = LLRThresholder()
        thresholder.output_type = "invalid"
        x = torch.tensor([0.5])
        with pytest.raises(ValueError, match="Unsupported output_type"):
            thresholder(x)


class TestMinDistanceThresholder:
    """Test MinDistanceThresholder class."""

    def test_init_default_probability(self):
        """Test MinDistanceThresholder initialization with default probability reference points."""
        thresholder = MinDistanceThresholder(input_type=InputType.PROBABILITY)
        expected_points = torch.tensor([0.0, 1.0])
        torch.testing.assert_close(thresholder.ref_points, expected_points)

    def test_init_default_llr(self):
        """Test MinDistanceThresholder initialization with default LLR reference points."""
        thresholder = MinDistanceThresholder(input_type=InputType.LLR)
        expected_points = torch.tensor([-2.0, 2.0])
        torch.testing.assert_close(thresholder.ref_points, expected_points)

    def test_init_custom_reference_points(self):
        """Test MinDistanceThresholder initialization with custom reference points."""
        custom_points = torch.tensor([0.2, 0.8])
        thresholder = MinDistanceThresholder(reference_points=custom_points)
        torch.testing.assert_close(thresholder.ref_points, custom_points)

    def test_forward_probability(self):
        """Test forward pass with probability inputs."""
        thresholder = MinDistanceThresholder(input_type=InputType.PROBABILITY)
        x = torch.tensor([0.1, 0.6, 0.2, 0.9])
        result = thresholder(x)
        # Should find closest reference point (0.0 or 1.0)
        expected = torch.tensor([0.0, 1.0, 0.0, 1.0])
        torch.testing.assert_close(result, expected)

    def test_forward_batch(self):
        """Test forward pass with batch inputs."""
        thresholder = MinDistanceThresholder()
        x = torch.tensor([[0.1, 0.9], [0.3, 0.7]])
        result = thresholder(x)
        expected = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        torch.testing.assert_close(result, expected)

    def test_forward_with_noise_var(self):
        """Test forward pass with custom noise variance."""
        thresholder = MinDistanceThresholder()
        x = torch.tensor([0.1, 0.9])
        result = thresholder(x, noise_var=0.5)
        # Should still work with custom noise variance
        assert result.shape == x.shape


class TestRepetitionSoftBitDecoder:
    """Test RepetitionSoftBitDecoder class."""

    def test_init_default(self):
        """Test RepetitionSoftBitDecoder initialization with default parameters."""
        decoder = RepetitionSoftBitDecoder()
        assert decoder.repetition_factor == 3
        assert decoder.soft_combine_method == "mean"
        assert isinstance(decoder.thresholder, FixedThresholder)

    def test_init_custom(self):
        """Test RepetitionSoftBitDecoder initialization with custom parameters."""
        custom_thresholder = FixedThresholder(threshold=0.3)
        decoder = RepetitionSoftBitDecoder(repetition_factor=5, soft_combine_method="median", thresholder=custom_thresholder)
        assert decoder.repetition_factor == 5
        assert decoder.soft_combine_method == "median"
        assert decoder.thresholder == custom_thresholder

    def test_init_invalid_repetition_factor(self):
        """Test error handling for invalid repetition factor."""
        with pytest.raises(ValueError, match="Repetition factor must be a positive integer"):
            RepetitionSoftBitDecoder(repetition_factor=0)

    def test_init_invalid_combine_method(self):
        """Test error handling for invalid combine method."""
        with pytest.raises(ValueError, match="Combine method must be one of"):
            RepetitionSoftBitDecoder(soft_combine_method="invalid")

    def test_init_llr_input_type(self):
        """Test initialization with LLR input type."""
        decoder = RepetitionSoftBitDecoder(input_type=InputType.LLR)
        assert isinstance(decoder.thresholder, LLRThresholder)

    def test_forward_mean_combination(self):
        """Test forward pass with mean combination."""
        decoder = RepetitionSoftBitDecoder(repetition_factor=3, soft_combine_method="mean")
        # Each bit repeated 3 times: [0.1, 0.2, 0.1] -> mean=0.13 -> 0, [0.8, 0.9, 0.7] -> mean=0.8 -> 1
        x = torch.tensor([[0.1, 0.2, 0.1, 0.8, 0.9, 0.7]])
        result = decoder(x)
        expected = torch.tensor([[0.0, 1.0]])
        torch.testing.assert_close(result, expected)

    def test_forward_sum_combination(self):
        """Test forward pass with sum combination."""
        decoder = RepetitionSoftBitDecoder(repetition_factor=3, soft_combine_method="sum")
        x = torch.tensor([[0.1, 0.2, 0.1, 0.3, 0.3, 0.3]])  # sums: 0.4, 0.9
        result = decoder(x)
        expected = torch.tensor([[0.0, 1.0]])
        torch.testing.assert_close(result, expected)

    def test_forward_median_combination(self):
        """Test forward pass with median combination."""
        decoder = RepetitionSoftBitDecoder(repetition_factor=3, soft_combine_method="median")
        x = torch.tensor([[0.1, 0.9, 0.2, 0.7, 0.8, 0.6]])  # medians: 0.2, 0.7
        result = decoder(x)
        expected = torch.tensor([[0.0, 1.0]])
        torch.testing.assert_close(result, expected)

    def test_forward_max_combination(self):
        """Test forward pass with max combination."""
        decoder = RepetitionSoftBitDecoder(repetition_factor=3, soft_combine_method="max")
        x = torch.tensor([[0.1, 0.3, 0.2, 0.4, 0.8, 0.6]])  # maxes: 0.3, 0.8
        result = decoder(x)
        expected = torch.tensor([[0.0, 1.0]])
        torch.testing.assert_close(result, expected)

    def test_forward_min_combination(self):
        """Test forward pass with min combination."""
        decoder = RepetitionSoftBitDecoder(repetition_factor=3, soft_combine_method="min")
        x = torch.tensor([[0.3, 0.6, 0.4, 0.7, 0.8, 0.9]])  # mins: 0.3, 0.7
        result = decoder(x)
        expected = torch.tensor([[0.0, 1.0]])
        torch.testing.assert_close(result, expected)


class TestHysteresisThresholder:
    """Test HysteresisThresholder class."""

    def test_init_default(self):
        """Test HysteresisThresholder initialization with default parameters."""
        thresholder = HysteresisThresholder()
        assert thresholder.high_threshold == 0.6
        assert thresholder.low_threshold == 0.4
        assert thresholder.input_type == InputType.PROBABILITY

    def test_init_custom(self):
        """Test HysteresisThresholder initialization with custom parameters."""
        thresholder = HysteresisThresholder(high_threshold=0.8, low_threshold=0.2)
        assert thresholder.high_threshold == 0.8
        assert thresholder.low_threshold == 0.2

    def test_init_invalid_thresholds(self):
        """Test error handling for invalid threshold values."""
        with pytest.raises(ValueError, match="high_threshold .* must be >= low_threshold"):
            HysteresisThresholder(high_threshold=0.3, low_threshold=0.7)

    def test_forward_hysteresis_effect(self):
        """Test the hysteresis effect."""
        thresholder = HysteresisThresholder(high_threshold=0.7, low_threshold=0.3)

        # First input - above high threshold
        x1 = torch.tensor([0.8])
        result1 = thresholder(x1)
        expected1 = torch.tensor([1.0])
        torch.testing.assert_close(result1, expected1)

        # Second input - between thresholds (should maintain previous state)
        x2 = torch.tensor([0.5])
        result2 = thresholder(x2)
        expected2 = torch.tensor([1.0])  # Should maintain high state
        torch.testing.assert_close(result2, expected2)

        # Third input - below low threshold
        x3 = torch.tensor([0.2])
        result3 = thresholder(x3)
        expected3 = torch.tensor([0.0])
        torch.testing.assert_close(result3, expected3)

    def test_reset_state(self):
        """Test state reset functionality."""
        thresholder = HysteresisThresholder()

        # Set initial state
        x1 = torch.tensor([0.8])
        thresholder(x1)

        # Reset and test
        x2 = torch.tensor([0.5])
        result = thresholder(x2, reset_state=True)
        # After reset, intermediate value should give 0 (starts from 0 state)
        expected = torch.tensor([0.0])
        torch.testing.assert_close(result, expected)

    def test_forward_llr_input(self):
        """Test forward pass with LLR input."""
        thresholder = HysteresisThresholder(high_threshold=1.0, low_threshold=-1.0, input_type=InputType.LLR)
        x = torch.tensor([2.0])  # High LLR
        result = thresholder(x)
        assert result.shape == x.shape


class TestWeightedThresholder:
    """Test WeightedThresholder class."""

    def test_init_scalar_weight(self):
        """Test WeightedThresholder initialization with scalar weight."""
        thresholder = WeightedThresholder(weights=0.8)
        assert thresholder.weights.numel() == 1
        assert thresholder.threshold == 0.5

    def test_init_list_weights(self):
        """Test WeightedThresholder initialization with list weights."""
        weights = [1.0, 0.8, 0.6]
        thresholder = WeightedThresholder(weights=weights)
        expected = torch.tensor(weights)
        torch.testing.assert_close(thresholder.weights, expected)

    def test_init_tensor_weights(self):
        """Test WeightedThresholder initialization with tensor weights."""
        weights = torch.tensor([1.0, 0.5])
        thresholder = WeightedThresholder(weights=weights)
        torch.testing.assert_close(thresholder.weights, weights)

    def test_init_normalize_weights(self):
        """Test weight normalization."""
        weights = [2.0, 4.0, 6.0]
        thresholder = WeightedThresholder(weights=weights, normalize_weights=True)
        expected = torch.tensor([2.0, 4.0, 6.0]) / 12.0  # Sum = 12
        torch.testing.assert_close(thresholder.weights, expected)

    def test_forward_scalar_weight(self):
        """Test forward pass with scalar weight."""
        thresholder = WeightedThresholder(weights=0.5, threshold=0.3)
        x = torch.tensor([0.8, 0.4])  # After weighting: [0.4, 0.2]
        result = thresholder(x)
        expected = torch.tensor([1.0, 0.0])  # 0.4 > 0.3, 0.2 < 0.3
        torch.testing.assert_close(result, expected)

    def test_forward_vector_weights(self):
        """Test forward pass with vector weights."""
        thresholder = WeightedThresholder(weights=[1.0, 0.5], threshold=0.4)
        x = torch.tensor([[0.5, 0.8]])  # After weighting: [0.5, 0.4]
        result = thresholder(x)
        expected = torch.tensor([[1.0, 0.0]])  # 0.5 > 0.4, 0.4 = 0.4 (not >)
        torch.testing.assert_close(result, expected)

    def test_forward_llr_input(self):
        """Test forward pass with LLR input."""
        thresholder = WeightedThresholder(weights=1.0, input_type=InputType.LLR)
        x = torch.tensor([-1.0, 1.0])
        result = thresholder(x)
        assert result.shape == x.shape


class TestSoftBitEnsembleThresholder:
    """Test SoftBitEnsembleThresholder class."""

    def test_init_majority_voting(self):
        """Test SoftBitEnsembleThresholder initialization with majority voting."""
        thresholders = [FixedThresholder(threshold=0.3), FixedThresholder(threshold=0.5), FixedThresholder(threshold=0.7)]
        ensemble = SoftBitEnsembleThresholder(thresholders, voting="majority")
        assert ensemble.voting == "majority"
        assert len(ensemble.thresholders) == 3

    def test_init_weighted_voting(self):
        """Test SoftBitEnsembleThresholder initialization with weighted voting."""
        thresholders = [FixedThresholder(), FixedThresholder()]
        weights = [0.7, 0.3]
        ensemble = SoftBitEnsembleThresholder(thresholders, voting="weighted", weights=weights)
        expected_weights = torch.tensor(weights)
        torch.testing.assert_close(ensemble.weights, expected_weights)

    def test_init_invalid_voting(self):
        """Test error handling for invalid voting method."""
        thresholders = [FixedThresholder()]
        with pytest.raises(ValueError, match="Voting method must be one of"):
            SoftBitEnsembleThresholder(thresholders, voting="invalid")

    def test_init_weights_mismatch(self):
        """Test error handling for weights mismatch."""
        thresholders = [FixedThresholder(), FixedThresholder()]
        weights = [0.5]  # Wrong number of weights
        with pytest.raises(ValueError, match="Number of weights .* must match"):
            SoftBitEnsembleThresholder(thresholders, voting="weighted", weights=weights)

    def test_forward_majority_voting(self):
        """Test forward pass with majority voting."""
        thresholders = [FixedThresholder(threshold=0.3), FixedThresholder(threshold=0.5), FixedThresholder(threshold=0.7)]  # Will output [1, 1, 1, 0]  # Will output [0, 1, 1, 0]  # Will output [0, 0, 1, 0]
        ensemble = SoftBitEnsembleThresholder(thresholders, voting="majority")
        x = torch.tensor([0.4, 0.6, 0.8, 0.2])
        result = ensemble(x)
        # Majority votes: [1/3, 2/3, 3/3, 0/3] -> [0, 1, 1, 0]
        expected = torch.tensor([0.0, 1.0, 1.0, 0.0])
        torch.testing.assert_close(result, expected)

    def test_forward_weighted_voting(self):
        """Test forward pass with weighted voting."""
        thresholders = [FixedThresholder(threshold=0.3), FixedThresholder(threshold=0.7)]
        weights = [0.3, 0.7]
        ensemble = SoftBitEnsembleThresholder(thresholders, voting="weighted", weights=weights)
        x = torch.tensor([0.5])  # First: 1, Second: 0 -> weighted: 0.3
        result = ensemble(x)
        expected = torch.tensor([0.0])  # 0.3 < 0.5
        torch.testing.assert_close(result, expected)

    def test_forward_any_voting(self):
        """Test forward pass with 'any' voting."""
        thresholders = [FixedThresholder(threshold=0.8), FixedThresholder(threshold=0.3)]  # Will output [0, 0]  # Will output [1, 1]
        ensemble = SoftBitEnsembleThresholder(thresholders, voting="any")
        x = torch.tensor([0.5, 0.9])
        result = ensemble(x)
        expected = torch.tensor([1.0, 1.0])  # Any thresholder outputs 1
        torch.testing.assert_close(result, expected)

    def test_forward_all_voting(self):
        """Test forward pass with 'all' voting."""
        thresholders = [FixedThresholder(threshold=0.3), FixedThresholder(threshold=0.7)]  # Will output [1, 1]  # Will output [0, 1]
        ensemble = SoftBitEnsembleThresholder(thresholders, voting="all")
        x = torch.tensor([0.5, 0.9])
        result = ensemble(x)
        expected = torch.tensor([0.0, 1.0])  # All must output 1
        torch.testing.assert_close(result, expected)


class TestDynamicThresholder:
    """Test DynamicThresholder class."""

    def test_init_default(self):
        """Test DynamicThresholder initialization with default parameters."""
        thresholder = DynamicThresholder()
        assert thresholder.decay == 0.9
        assert thresholder.threshold == 0.5
        assert thresholder.adaptation_method == "mean"

    def test_init_custom(self):
        """Test DynamicThresholder initialization with custom parameters."""
        thresholder = DynamicThresholder(decay=0.8, initial_threshold=0.3, adaptation_method="median", bias=0.1)
        assert thresholder.decay == 0.8
        assert thresholder.threshold == 0.3
        assert thresholder.adaptation_method == "median"
        assert thresholder.bias == 0.1

    def test_init_invalid_decay(self):
        """Test error handling for invalid decay value."""
        with pytest.raises(ValueError, match="Decay must be between 0 and 1"):
            DynamicThresholder(decay=1.5)

    def test_init_invalid_adaptation_method(self):
        """Test error handling for invalid adaptation method."""
        with pytest.raises(ValueError, match="Adaptation method must be one of"):
            DynamicThresholder(adaptation_method="invalid")

    def test_forward_mean_adaptation(self):
        """Test forward pass with mean adaptation."""
        thresholder = DynamicThresholder(adaptation_method="mean", decay=0.5)

        # First batch
        x1 = torch.tensor([0.1, 0.2, 0.8, 0.9])  # mean = 0.5
        result1 = thresholder(x1)

        # Second batch with different mean
        x2 = torch.tensor([0.2, 0.3, 0.7, 0.8])  # mean = 0.5
        result2 = thresholder(x2)

        # Threshold should adapt
        assert result1.shape == x1.shape
        assert result2.shape == x2.shape

    def test_forward_median_adaptation(self):
        """Test forward pass with median adaptation."""
        thresholder = DynamicThresholder(adaptation_method="median")
        x = torch.tensor([0.1, 0.4, 0.6, 0.9])
        result = thresholder(x)
        assert result.shape == x.shape

    def test_forward_percentile_adaptation(self):
        """Test forward pass with percentile adaptation."""
        thresholder = DynamicThresholder(adaptation_method="percentile")
        x = torch.tensor([0.1, 0.3, 0.7, 0.9])
        result = thresholder(x, percentile=75.0)
        assert result.shape == x.shape

    def test_reset_stats(self):
        """Test statistics reset functionality."""
        thresholder = DynamicThresholder()

        # Run some data to change statistics
        x1 = torch.tensor([0.8, 0.9])
        thresholder(x1)

        # Reset statistics
        thresholder.reset_stats(initial_threshold=0.3)
        assert thresholder.threshold == 0.3
        assert thresholder.running_mean == 0.3

    def test_forward_with_reset(self):
        """Test forward pass with reset flag."""
        thresholder = DynamicThresholder()
        x = torch.tensor([0.5, 0.6])
        result = thresholder(x, reset=True)
        assert result.shape == x.shape

    def test_forward_llr_input(self):
        """Test forward pass with LLR input."""
        thresholder = DynamicThresholder(input_type=InputType.LLR)
        x = torch.tensor([-1.0, 0.5, 1.0])
        result = thresholder(x)
        assert result.shape == x.shape

    def test_threshold_limits(self):
        """Test threshold limiting functionality."""
        thresholder = DynamicThresholder(min_threshold=0.2, max_threshold=0.8, adaptation_method="mean")

        # Test with very high mean
        x_high = torch.full((100,), 0.95)
        thresholder(x_high)
        assert thresholder.threshold <= 0.8

        # Reset and test with very low mean
        thresholder.reset_stats()
        x_low = torch.full((100,), 0.05)
        thresholder(x_low)
        assert thresholder.threshold >= 0.2


class TestSoftBitThresholderBase:
    """Test SoftBitThresholder base class."""

    def test_to_device(self):
        """Test moving thresholder to device."""
        thresholder = FixedThresholder()
        device = torch.device("cpu")
        result = thresholder.to(device)
        assert result.device == device
        assert result is thresholder  # Should return self for chaining

    def test_abstract_forward(self):
        """Test that base class forward is abstract."""
        base_thresholder = SoftBitThresholder()
        x = torch.tensor([0.5])
        with pytest.raises(NotImplementedError):
            base_thresholder(x)
