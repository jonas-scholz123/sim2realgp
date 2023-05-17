from torch import nn
import torch


class FiLM(nn.Module):
    def __init__(self, n_features) -> None:
        super().__init__()
        self.scales = nn.parameter.Parameter(torch.zeros(n_features) + 1)
        self.biases = nn.parameter.Parameter(torch.zeros(n_features))

    def forward(self, x: torch.Tensor):
        return self.scales[None, :, None] * x + self.biases[None, :, None]
