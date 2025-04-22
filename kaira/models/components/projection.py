"""Projection components for dimensionality reduction in communication systems.

This module implements projection methods that can be used for dimensionality reduction in
communication systems, as described and implemented in
:cite:`yilmaz2025learning,yilmaz2025private`.
"""

from enum import Enum
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from kaira.models.base import BaseModel

from ..registry import ModelRegistry


class ProjectionType(Enum):
    """Enum for different types of projections."""

    #: Random binary projection matrix with values {-1, 1}.
    #: Suitable for fast computation and memory-efficient implementations.
    RADEMACHER = "rademacher"

    #: Random projection matrix with values drawn from N(0, 1/out_features).
    #: Provides good theoretical guarantees for dimensionality reduction.
    GAUSSIAN = "gaussian"

    #: Random orthogonal matrix with columns that form an orthonormal basis.
    #: Preserves angles and distances better than non-orthogonal projections.
    ORTHOGONAL = "orthogonal"

    #: Complex-valued projection with real and imaginary parts from N(0, 1/(2*out_features)).
    #: Useful for complex signal processing and wireless communications.
    COMPLEX_GAUSSIAN = "complex_gaussian"

    #: Complex-valued orthogonal projection with orthonormal columns.
    #: Provides optimal preservation of signal characteristics for complex data.
    COMPLEX_ORTHOGONAL = "complex_orthogonal"


@ModelRegistry.register_model()
class Projection(BaseModel):
    """Projection layer for dimensionality reduction in communication systems
    :cite:`yilmaz2025private,yilmaz2025learning`.

    This module implements different projection methods that can be used for dimensionality
    reduction in communication systems. These projection methods have been adapted from those
    used in :cite:`yilmaz2025private` and :cite:`yilmaz2025learning`. The projection only operates
    on the last dimension of the input tensor and uses matrix multiplication.

    Available projection types:
    * RADEMACHER: Random matrix with values {-1, 1} (binary)
    * GAUSSIAN: Random matrix with values from N(0, 1/out_features)
    * ORTHOGONAL: Matrix with orthogonal columns (real-valued)
    * COMPLEX_GAUSSIAN: Complex matrix with real and imaginary parts from N(0, 1/(2*out_features))
    * COMPLEX_ORTHOGONAL: Complex matrix with orthogonal columns

    Complex projections are particularly useful for wireless communication systems
    where signals are often represented in the complex domain with I/Q components.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        projection_type: Union[ProjectionType, str] = ProjectionType.ORTHOGONAL,
        seed: Optional[int] = None,
        trainable: bool = True,
        dtype: Optional[torch.dtype] = None,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize the Projection layer.

        Args:
            in_features (int): The dimensionality of the input features.
            out_features (int): The dimensionality of the output features.
            projection_type (ProjectionType or str, optional): Type of projection to use.
                Possible values as enum: ProjectionType.RADEMACHER, ProjectionType.GAUSSIAN,
                ProjectionType.ORTHOGONAL, ProjectionType.COMPLEX_GAUSSIAN, ProjectionType.COMPLEX_ORTHOGONAL.
                Possible values as str: "rademacher", "gaussian", "orthogonal", "complex_gaussian", "complex_orthogonal".
                Default is ProjectionType.ORTHOGONAL.
            seed (int, optional): Random seed for reproducibility. Default is None.
            trainable (bool, optional): Whether the projection matrix is trainable.
                Default is True.
            dtype (torch.dtype, optional): The dtype of the projection matrix.
                Default is None, which will use float32 for real projections and complex64
                for complex projections.
            *args: Variable positional arguments passed to the base class.
            **kwargs: Variable keyword arguments passed to the base class.
        """
        super().__init__(*args, **kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.projection_type = projection_type if isinstance(projection_type, ProjectionType) else ProjectionType(projection_type)
        self.seed = seed
        self.trainable = trainable

        # Determine if we're using a complex projection
        self.is_complex = self.projection_type in [ProjectionType.COMPLEX_GAUSSIAN, ProjectionType.COMPLEX_ORTHOGONAL]

        # Determine dtype based on input or defaults
        if dtype is None:
            self.dtype = torch.complex64 if self.is_complex else torch.float32
        else:
            self.dtype = dtype

        # Create local RNG for PyTorch
        torch_rng = torch.Generator()

        # Set seed for the local RNG if provided
        if seed is not None:
            torch_rng.manual_seed(seed)

        # Initialize projection matrix based on the specified type
        if self.projection_type == ProjectionType.RADEMACHER:
            # Rademacher distribution: Random matrix with values {-1, 1}
            projection = (torch.randint(0, 2, (in_features, out_features), generator=torch_rng) * 2 - 1).to(self.dtype)
        elif self.projection_type == ProjectionType.GAUSSIAN:
            # Gaussian distribution: Random matrix with values from N(0, 1/out_features)
            projection = (torch.randn(in_features, out_features, generator=torch_rng) / torch.sqrt(torch.tensor(out_features, dtype=torch.float32))).to(self.dtype)
        elif self.projection_type == ProjectionType.ORTHOGONAL:
            # Orthogonal matrix: Using QR decomposition for orthogonal initialization
            random_matrix = torch.randn(max(in_features, out_features), min(in_features, out_features), generator=torch_rng)
            q, r = torch.linalg.qr(random_matrix)
            # Use the sign of diagonal elements of r to ensure deterministic results
            d = torch.diagonal(r)
            ph = d.sign()
            q *= ph

            if in_features >= out_features:
                projection = q[:in_features, :out_features].to(self.dtype)
            else:
                projection = q[:in_features, :out_features].t().to(self.dtype)
        elif self.projection_type == ProjectionType.COMPLEX_GAUSSIAN:
            # Complex Gaussian: Real and imaginary parts from N(0, 1/(2*out_features))
            # Factor of 1/2 ensures same expected power as real Gaussian
            real_part = torch.randn(in_features, out_features, generator=torch_rng) / torch.sqrt(torch.tensor(2 * out_features, dtype=torch.float32))
            imag_part = torch.randn(in_features, out_features, generator=torch_rng) / torch.sqrt(torch.tensor(2 * out_features, dtype=torch.float32))
            projection = torch.complex(real_part, imag_part).to(self.dtype)
        elif self.projection_type == ProjectionType.COMPLEX_ORTHOGONAL:
            # Complex Orthogonal matrix: Generate a random complex matrix and orthogonalize
            real_part = torch.randn(max(in_features, out_features), min(in_features, out_features), generator=torch_rng)
            imag_part = torch.randn(max(in_features, out_features), min(in_features, out_features), generator=torch_rng)
            random_matrix = torch.complex(real_part, imag_part)

            # Use QR decomposition to get an orthogonal basis
            q, r = torch.linalg.qr(random_matrix)

            # Normalize phases to ensure deterministic results
            d = torch.diagonal(r)
            ph = d / torch.abs(d)  # Unit complex numbers preserving phase
            q *= ph

            if in_features >= out_features:
                projection = q[:in_features, :out_features].to(self.dtype)
            else:
                projection = q[:in_features, :out_features].t().to(self.dtype)
        else:
            raise ValueError(f"Unknown projection type: {projection_type}")

        # Register the projection matrix as a parameter or buffer
        if trainable:
            self.projection = nn.Parameter(projection)
        else:
            self.register_buffer("projection", projection)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass of the Projection layer.

        Args:
            x (torch.Tensor): Input tensor with the last dimension being the features.
                For complex projections, x can be either a complex tensor or a real tensor.
                If x is real and the projection is complex, x will be treated as having
                only real components.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: Output tensor with the last dimension projected.
                If the projection is complex, the output will be complex.
        """
        # Handle type conversions if needed
        if self.is_complex and not torch.is_complex(x):
            # Real input with complex projection - treat input as having only real part
            x = torch.complex(x, torch.zeros_like(x))

        # Perform matrix multiplication on the last dimension
        return x @ self.projection

    def extra_repr(self) -> str:
        """Return extra representation string for the module."""
        return f"in_features={self.in_features}, out_features={self.out_features}, " f"projection_type={self.projection_type.value}, is_complex={self.is_complex}, " f"dtype={self.dtype}, trainable={self.trainable}"
