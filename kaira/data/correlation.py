"""Correlation models for data generation and simulation.

This module contains models for simulating statistical correlations between data sources, which is
particularly useful for distributed source coding scenarios.
"""

from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset

from kaira.models.wyner_ziv import WynerZivCorrelationModel


class WynerZivCorrelationDataset(Dataset):
    r"""Dataset for Wyner-Ziv coding scenarios with correlated sources.

    This dataset pairs source data with correlated side information according to a
    specified correlation model. It's particularly useful for simulating and evaluating
    Wyner-Ziv coding scenarios where the decoder has access to side information that is
    statistically correlated with the source.

    Attributes:
        model: The correlation model used to generate side information
        data: The source data tensor with shape (n_samples, \*feature_dims)
        correlated_data: The correlated side information with same shape as source data
    """

    def __init__(self, source: torch.Tensor, correlation_type: str = "gaussian", correlation_params: Optional[Dict[str, Any]] = None, *args, **kwargs):
        """Initialize the Wyner-Ziv correlated dataset.

        Args:
            source: Source data tensor where the first dimension represents the number of samples
            correlation_type: Type of correlation model:
                - 'gaussian': Additive Gaussian noise
                - 'binary': Binary symmetric channel
                - 'custom': User-defined model
            correlation_params: Parameters for the correlation model:
                - For 'gaussian': {'sigma': float} - Standard deviation of the noise
                - For 'binary': {'crossover_prob': float} - Probability of bit flipping
                - For 'custom': {'transform_fn': callable} - Custom transformation function
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)  # Pass args and kwargs to parent if necessary
        self.model = WynerZivCorrelationModel(correlation_type, correlation_params, *args, **kwargs)
        self.data = source
        self.correlated_data = self.model(source, *args, **kwargs)

    def __len__(self):
        """Return the number of samples in the dataset.

        Returns:
            int: The number of samples, corresponding to the first dimension of data
        """
        return self.data.size(0)

    def __getitem__(self, idx):
        """Retrieve a source-side information pair from the dataset at the specified index.

        Args:
            idx: Index or slice object to index into the dataset

        Returns:
            tuple: A pair of tensors (source, side_information) representing the
                  source data and its correlated side information at the specified
                  index/indices
        """
        return self.data[idx], self.correlated_data[idx]
