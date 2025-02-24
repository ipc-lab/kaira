import torch

from kaira.core import BaseConstraint

__all__ = [
    "TotalPowerConstraint",
    "AveragePowerConstraint",
    "ComplexTotalPowerConstraint",
    "ComplexAveragePowerConstraint",
]


class TotalPowerConstraint(BaseConstraint):
    def __init__(self, total_power: float) -> None:
        super().__init__()
        self.total_power = total_power
        self.total_power_factor = torch.sqrt(torch.tensor(self.total_power))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = torch.norm(x, dim=tuple(range(1, len(x.shape))), keepdim=True)
        x = x * self.total_power_factor / x_norm
        return x


class AveragePowerConstraint(BaseConstraint):
    def __init__(self, average_power: float) -> None:
        super().__init__()
        self.average_power = average_power
        self.power_avg_factor = torch.sqrt(torch.tensor(self.average_power))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = torch.norm(x, dim=tuple(range(1, len(x.shape))), keepdim=True)
        avg_power_sqrt = self.power_avg_factor * torch.sqrt(
            torch.prod(torch.tensor(x.shape[1:]), 0)
        )
        x = x * self.power_avg_factor * avg_power_sqrt / x_norm
        return x


class ComplexTotalPowerConstraint(TotalPowerConstraint):
    def __init__(self, total_power: float) -> None:
        super().__init__(total_power * torch.sqrt(0.5))


class ComplexAveragePowerConstraint(AveragePowerConstraint):
    def __init__(self, average_power: float) -> None:
        super().__init__(average_power * torch.sqrt(0.5))
