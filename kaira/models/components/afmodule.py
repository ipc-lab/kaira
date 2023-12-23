import torch
from torch import nn


class AFModule(nn.Module):
    def __init__(self, N, csi_length):
        super().__init__()

        self.c_in = N

        self.layers = nn.Sequential(
            nn.Linear(in_features=N + csi_length, out_features=N),
            nn.LeakyReLU(),
            nn.Linear(in_features=N, out_features=N),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x, side_info = x
        context = torch.mean(x, dim=(2, 3))

        context_input = torch.cat([context, side_info], dim=1)
        mask = self.layers(context_input).view(-1, self.c_in, 1, 1)

        out = mask * x
        return out
