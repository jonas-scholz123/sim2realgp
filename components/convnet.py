import math
from typing import Optional
from neuralprocesses.util import compress_batch_dimensions, with_first_last
import neuralprocesses.torch as nps
import torch

class ConvNet(nps.Module):
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
        activation = torch.nn.ReLU()

        self.conv_net = torch.nn.Sequential(
            *(
                nps.Conv(
                    dim=dim,
                    in_channels=in_channels if first else channels,
                    out_channels=out_channels if last else channels,
                    kernel=kernel,
                    activation=None if first else activation,
                    separable=separable,
                    residual=residual,
                    dtype=dtype,
                )
                for first, last, _ in with_first_last(range(num_layers))
            )
        )

    def __call__(self, x):
        x, uncompress = compress_batch_dimensions(x, self.dim + 1)
        return uncompress(self.conv_net(x))