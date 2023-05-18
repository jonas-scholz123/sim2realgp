from torch import nn
import torch


class TransposeConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        super().__init__()

        # TODO: does this if do anything?
        if stride > 1:
            output_padding = stride // 2

        padding = kernel_size // 2

        self.transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.transpose(x)
        return h
