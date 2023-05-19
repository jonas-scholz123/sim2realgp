from torch import nn
import torch
import lab as B


class FiLM(nn.Module):
    def __init__(self, n_features, freeze_affine: bool = True) -> None:
        super().__init__()
        device = B.ActiveDevice.active_name
        self.scales = nn.parameter.Parameter(torch.zeros(n_features, device=device) + 1)
        self.biases = nn.parameter.Parameter(torch.zeros(n_features, device=device))

        # Freeze film layer during pretraining if wanted.
        self.requires_grad_(not freeze_affine)

    def forward(self, x: torch.Tensor):
        return self.scales[None, :, None] * x + self.biases[None, :, None]
