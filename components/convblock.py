import torch

from components.film import FiLM


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
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel, stride=1, padding=padding
        )

        self.residual = residual

        if affine:
            self.affine = FiLM(in_channels)
        else:
            self.affine = nn.Identity()

        if residual:
            self.residual = (
                torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)
                if in_channels != out_channels
                else torch.nn.Identity()
            )

    def forward(self, x: torch.Tensor):
        h = self.conv(self.activation(self.affine(x)))
        if self.residual:
            return h + self.residual(x)
        else:
            return h


class DoubleConvBlock(torch.nn.Module):
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
        h = self.conv1(self.activation(self.affine1(x)))
        h = self.conv2(self.activation(self.affine2(h)))
        if self.residual:
            return h + self.residual(x)
        else:
            return h
