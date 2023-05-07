# %%
import math
from typing import Optional, Iterable
from neuralprocesses.util import compress_batch_dimensions, with_first_last
import neuralprocesses.torch as nps
import torch
from torch import nn


class ConvNet_(torch.nn.Module):
    """A regular convolutional neural network.

    Args:
        dim (int): Dimensionality.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        channels (int): Number of channels at every intermediate layer.
        num_layers (int): Number of layers.
        points_per_unit (float, optional): Density of the discretisation corresponding
            to the inputs.
        receptive_field (float, optional): Desired receptive field.
        kernel (int, optional): Kernel size. If set, then this overrides the computation
            done by `points_per_unit` and `receptive_field`.
        separable (bool, optional): Use depthwise separable convolutions. Defaults
            to `True`.
        dtype (dtype, optional): Data type.

    Attributes:
        dim (int): Dimensionality.
        kernel (int): Kernel size.
        num_halving_layers (int): Number of layers with stride equal to two.
        receptive_field (float): Receptive field.
        conv_net (module): The architecture.
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        channels: int,
        num_layers: int,
        kernel: Optional[int] = None,
        points_per_unit: Optional[float] = 1,
        receptive_field: Optional[float] = None,
        separable: bool = True,
        residual: bool = False,
        dtype=None,
    ):
        super().__init__()
        self.dim = dim

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

        # Make it a drop-in substitute for :class:`UNet`.
        self.num_halving_layers = 0
        self.receptive_field = receptive_field

        # Construct basic building blocks.
        self.activation = torch.nn.ReLU()

        self.conv_layers = torch.nn.ModuleList(
            [
                nps.Conv(
                    dim=dim,
                    in_channels=in_channels if first else channels,
                    out_channels=out_channels if last else channels,
                    kernel=kernel,
                    # activation=self.activation if not first else None,
                    separable=separable,
                    residual=False,
                    dtype=dtype,
                )
                for first, last, _ in with_first_last(range(num_layers))
            ]
        )

        self.batchnorms = torch.nn.ModuleList(
            [
                torch.nn.BatchNorm1d(channels, affine=False)
                for _ in range(num_layers - 2)
            ]
        )

        # self.bn = torch.nn.BatchNorm1d(channels, affine=False)

        # layers = []

        # for first, last, _ in with_first_last(range(num_layers)):
        #    if first:
        #        layers.append(
        #            nps.Conv(
        #                dim=dim,
        #                in_channels=in_channels,
        #                out_channels=channels,
        #                kernel=kernel,
        #                activation=None,
        #                dtype=dtype,
        #            )
        #        )
        #        layers.append(torch.nn.ReLU())
        #        continue

        #    if last:
        #        layers.append(
        #            nps.Conv(
        #                dim=dim,
        #                in_channels=channels,
        #                out_channels=out_channels,
        #                kernel=kernel,
        #                activation=None,
        #                dtype=dtype,
        #            )
        #        )
        #        layers.append(torch.nn.ReLU())
        #        continue

        #    layers.append(
        #        nps.ResidualBlock(
        #            layer1=lambda x: x,
        #            layer2=nps.Conv(
        #                dim=dim,
        #                in_channels=channels,
        #                out_channels=channels,
        #                kernel=kernel,
        #                activation=torch.nn.ReLU(),
        #                dtype=dtype,
        #            ),
        #            layer_post=lambda x: x,
        #        )
        #    )

        ## print(torch.nn.Sequential(*layers))
        # self.conv_net = nps.Sequential(*layers)

        # layers = [
        #    nps.ResidualBlock(
        #        lambda x: x,
        #        nps.Conv(
        #            dim=dim,
        #            in_channels=in_channels if first else channels,
        #            out_channels=out_channels if last else channels,
        #            kernel=kernel,
        #            activation=self.activation if not first else None,
        #            separable=separable,
        #            residual=True,
        #            dtype=dtype,
        #        ),
        #    )
        #    for first, last, _ in with_first_last(range(num_layers))
        # ]

        # self.batchnorms = torch.nn.ModuleList(
        # [torch.nn.BatchNorm1d(channels) for _ in range(num_layers - 2)])

    def forward(self, x):
        x, uncompress = compress_batch_dimensions(x, self.dim + 1)

        for i, conv_layer in enumerate(self.conv_layers):
            a = x

            if i not in (0, len(self.conv_layers) - 1):
                x = self.batchnorms[i - 1](x)

            x = self.activation(x)
            x = conv_layer(x)

            if i not in (0, len(self.conv_layers) - 1):
                # x = self.batchnorms[i - 1](x)
                x += a

            # if i not in (0, len(self.conv_layers) - 1):
            # x = self.activation(x)

        # x = self.conv_layers[-1](x)

        # conv_net = torch.nn.Sequential(*self.conv_layers)

        # return uncompress(self.conv_net(x))
        return uncompress(x)


#    def __call__(self, x):
#        x, uncompress = compress_batch_dimensions(x, self.dim + 1)
#
#        # x = self.conv_layers[0](x)
#
#        for i, conv_layer in enumerate(self.conv_layers):
#            middle = i not in (0, len(self.conv_layers) - 1)
#
#            original_x = x
#            x = self.activation(x)
#            if middle:
#                x = self.batchnorms[i - 1](x)
#            x = conv_layer(x)
#        # x = self.activation(x)
#        if middle:
#            # x = self.batchnorms[i - 1](x)
#            x += original_x
#
#        # x = self.conv_layers[-1](x)
#
#        # conv_net = torch.nn.Sequential(*self.conv_layers)
#
#        # return uncompress(self.conv_net(x))
#        return uncompress(x)


# class ConvNetF(torch.nn.Module):
#    def __init__(self) -> None:
#        super().__init__()


class ConvNet(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        channels: int,
        num_layers: int,
        kernel: Optional[int] = None,
        points_per_unit: Optional[float] = 1,
        receptive_field: Optional[float] = None,
        separable: bool = True,
        residual: bool = False,
        dtype=None,
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
                ResBlock(
                    in_channels=channels,
                    out_channels=channels,
                    kernel=self.kernel,
                    norm=True,
                    residual=False,
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


class ResBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        norm: bool = False,
        residual: bool = True,
    ) -> None:
        super().__init__()

        # print(f"{out_channels=}, {in_channels=}")
        # print(f"{kernel=}")

        padding = (kernel - 1) // 2

        self.activation = nn.ReLU()
        self.residual = residual

        if norm:
            self.norm1 = nn.BatchNorm1d(in_channels)
            self.norm2 = nn.BatchNorm1d(out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

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
        h = self.conv1(self.activation(self.norm1(x)))
        # print(f"{h.shape=}")
        h = self.conv2(self.activation(self.norm2(h)))
        # print(f"{h.shape=}")
        # print(self.residual(x).shape)
        if self.residual:
            return h + self.residual(x)
        else:
            return h


if __name__ == "__main__":
    model = ConvNet(1, 10, 10, 10, 5, None)
    print(model)
