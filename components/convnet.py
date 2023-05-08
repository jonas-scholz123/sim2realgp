# %%
import math
from typing import Optional, Iterable
from neuralprocesses.util import compress_batch_dimensions, with_first_last
import neuralprocesses.torch as nps
import torch
from torch import nn


class ConvNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: int,
        num_layers: int,
        kernel: Optional[int] = None,
        points_per_unit: Optional[float] = 1,
        receptive_field: Optional[float] = None,
        affine: bool = True,
        residual: bool = True,
    ):
        super().__init__()

        # Make it a drop-in substitute for :class:`UNet`.
        self.num_halving_layers = 0
        self.receptive_field = receptive_field
        self.dim = 1

        if kernel is None:
            # Compute kernel size.
            receptive_points = receptive_field * points_per_unit
            kernel = math.ceil(1 + (receptive_points - 1) / num_layers)
            kernel = kernel + 1 if kernel % 2 == 0 else kernel  # Make kernel size odd.
            self.kernel = kernel  # Store it for reference.
        else:
            # Compute the receptive field size.
            receptive_points = kernel + num_layers * (kernel - 1)
            receptive_field = receptive_points / points_per_unit
            self.kernel = kernel

        layers = [
            nn.Conv1d(
                in_channels, channels, self.kernel, padding=(self.kernel - 1) // 2
            ),
            nn.ReLU(),
            nn.Conv1d(channels, channels, self.kernel, padding=(self.kernel - 1) // 2),
            nn.ReLU(),
        ]

        layers.extend(
            [
                ConvBlock(
                    in_channels=channels,
                    out_channels=channels,
                    kernel=self.kernel,
                    affine=affine,
                    residual=residual,
                )
                for _ in range(num_layers)
            ]
        )

        layers.extend(
            [
                nn.Conv1d(
                    channels, channels, self.kernel, padding=(self.kernel - 1) // 2
                ),
                nn.ReLU(),
                nn.Conv1d(
                    channels, out_channels, self.kernel, padding=(self.kernel - 1) // 2
                ),
            ]
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x, uncompress = compress_batch_dimensions(x, self.dim + 1)
        return uncompress(self.net(x))


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        affine: bool = False,
        residual: bool = True,
    ) -> None:
        super().__init__()

        padding = (kernel - 1) // 2

        self.activation = nn.ReLU()
        self.residual = residual

        if affine:
            self.affine1 = FiLM(in_channels)
            self.affine2 = FiLM(out_channels)
        else:
            self.affine1 = nn.Identity()
            self.affine2 = nn.Identity()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel, stride=1, padding=padding
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel, stride=1, padding=padding
        )

        self.residual = (
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else torch.nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        # print(f"{x.shape=}")
        h = self.conv1(self.activation(self.affine1(x)))
        # print(f"{h.shape=}")
        h = self.conv2(self.activation(self.affine2(h)))
        # print(f"{h.shape=}")
        # print(self.residual(x).shape)
        if self.residual:
            return h + self.residual(x)
        else:
            return h


class FiLM(nn.Module):
    def __init__(self, n_features) -> None:
        super().__init__()
        self.scales = nn.parameter.Parameter(torch.zeros(n_features) + 1)
        self.biases = nn.parameter.Parameter(torch.zeros(n_features))

    def forward(self, x: torch.Tensor):
        return self.scales[None, :, None] * x + self.biases[None, :, None]


if __name__ == "__main__":
    model = ConvNet(2, 32, 64, 5, 30, affine=True, residual=True)
    # x = torch.rand(16, 2, 12)
    # model(x)

# model

# i = 0
# j = 4
# k = 0
# print(list(list(model.modules())[i].net[j].parameters())[k])
# print(list(list(list(model.modules())[i].net[j].modules())[0].modules())[k])

# list(model.parameters())
# [a.shape for a in list(model.parameters())]
