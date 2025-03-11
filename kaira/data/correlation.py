"""Correlation models for data generation and simulation.

This module contains models for simulating statistical correlations between data sources,
which is particularly useful for distributed source coding scenarios.
"""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class WynerZivCorrelationModel(nn.Module):
    """Model for simulating correlation between source and side information.

    In Wyner-Ziv coding, there is correlation between the source X and the side information
    Y available at the decoder. This module simulates different correlation models between
    the source X and the side information Y. The correlation structure is critical as it
    determines the theoretical rate bounds and practical coding efficiency.

    The correlation model effectively creates a virtual channel between X and Y, which
    can be modeled as various types of conditional probability distributions p(Y|X).

    Attributes:
        correlation_type (str): Type of correlation model ('gaussian', 'binary', 'custom')
            - 'gaussian': Additive white Gaussian noise model (Y = X + N, where N ~ N(0, σ²))
            - 'binary': Binary symmetric channel model with crossover probability p
            - 'custom': User-defined correlation model through a transform function
        correlation_params (Dict): Parameters specific to the correlation model
    """

    def __init__(
        self,
        correlation_type: str = "gaussian",
        correlation_params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the correlation model.

        Args:
            correlation_type: Type of correlation model:
                - 'gaussian': Additive Gaussian noise (requires 'sigma' parameter)
                - 'binary': Binary symmetric channel (requires 'crossover_prob' parameter)
                - 'custom': User-defined model (requires 'transform_fn' parameter)
            correlation_params: Parameters for the correlation model:
                - For 'gaussian': {'sigma': float} - Standard deviation of the noise
                - For 'binary': {'crossover_prob': float} - Probability of bit flipping
                - For 'custom': {'transform_fn': callable} - Custom transformation function
        """
        super().__init__()
        self.correlation_type = correlation_type
        self.correlation_params = correlation_params or {}

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        """Generate correlated side information from the source.

        Creates side information Y that is correlated with the source X according to
        the specified correlation model. This simulates the scenario where the decoder
        has access to side information that is statistically related to the source.

        Args:
            source: Source signal X (can be continuous or discrete valued)

        Returns:
            Correlated side information Y with statistical dependence on X according
            to the specified correlation model

        Raises:
            ValueError: If the correlation type is unknown or if the custom correlation
                model is missing the required transform function
        """
        if self.correlation_type == "gaussian":
            # Y = X + Z, where Z ~ N(0, sigma²)
            sigma = self.correlation_params.get("sigma", 1.0)
            noise = torch.randn_like(source) * sigma
            return source + noise

        elif self.correlation_type == "binary":
            # Binary symmetric channel with crossover probability p
            p = self.correlation_params.get("crossover_prob", 0.1)
            flip_mask = torch.bernoulli(torch.full_like(source, p))
            return source * (1 - flip_mask) + (1 - source) * flip_mask

        elif self.correlation_type == "custom":
            # Custom correlation model
            if "transform_fn" in self.correlation_params:
                return self.correlation_params["transform_fn"](source)
            else:
                raise ValueError("Custom correlation model requires 'transform_fn' parameter")

        else:
            raise ValueError(f"Unknown correlation type: {self.correlation_type}")


class WynerZivCorrelationDataset(Dataset):
    def __init__(self, source: torch.Tensor, correlation_type: str = "gaussian", correlation_params: Optional[Dict[str, Any]] = None):
        self.model = WynerZivCorrelationModel(correlation_type, correlation_params)
        self.data = source
        self.correlated_data = self.model(source)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], self.correlated_data[idx]