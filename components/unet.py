import math
from typing import Union, Tuple
from neuralprocesses.util import compress_batch_dimensions
import neuralprocesses.torch as nps
from torch import nn
from functools import partial
import lab as B

from components.convblock import ConvBlock


class UNet(nn.Module):
    """UNet.

    Args:
        dim (int): Dimensionality.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        channels (tuple[int], optional): Channels of every layer of the UNet.
            Defaults to six layers each with 64 channels.
        kernels (int or tuple[int], optional): Sizes of the kernels. Defaults to `5`.
        strides (int or tuple[int], optional): Strides. Defaults to `2`.
        activations (object or tuple[object], optional): Activation functions.
        separable (bool, optional): Use depthwise separable convolutions. Defaults to
            `False`.
        residual (bool, optional): Make residual convolutional blocks. Defaults to
            `False`.
        resize_convs (bool, optional): Use resize convolutions rather than
            transposed convolutions. Defaults to `False`.
        resize_conv_interp_method (str, optional): Interpolation method for the
            resize convolutions. Can be set to "bilinear". Defaults to "nearest".
        dtype (dtype, optional): Data type.

    Attributes:
        dim (int): Dimensionality.
        kernels (tuple[int]): Sizes of the kernels.
        strides (tuple[int]): Strides.
        activations (tuple[function]): Activation functions.
        num_halving_layers (int): Number of layers with stride equal to two.
        receptive_fields (list[float]): Receptive field for every intermediate value.
        receptive_field (float): Receptive field of the model.
        before_turn_layers (list[module]): Layers before the U-turn.
        after_turn_layers (list[module]): Layers after the U-turn
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        channels: Tuple[int, ...] = (8, 16, 16, 32, 32, 64),
        kernels: Union[int, Tuple[Union[int, Tuple[int, ...]], ...]] = 5,
        strides: Union[int, Tuple[int, ...]] = 2,
        activations: Union[None, object, Tuple[object, ...]] = None,
        separable: bool = False,
        residual: bool = False,
        resize_convs: bool = False,
        resize_conv_interp_method: str = "nearest",
        dtype=None,
    ):
        super().__init__()
        self.dim = dim

        # If `kernel` is an integer, repeat it for every layer.
        if not isinstance(kernels, (tuple, list)):
            kernels = (kernels,) * len(channels)
        elif len(kernels) != len(channels):
            raise ValueError(
                f"Length of `kernels` ({len(kernels)}) must equal "
                f"the length of `channels` ({len(channels)})."
            )
        self.kernels = kernels

        # If `strides` is an integer, repeat it for every layer.
        # TODO: Change the default so that the first stride is 1.
        if not isinstance(strides, (tuple, list)):
            strides = (strides,) * len(channels)
        elif len(strides) != len(channels):
            raise ValueError(
                f"Length of `strides` ({len(strides)}) must equal "
                f"the length of `channels` ({len(channels)})."
            )
        self.strides = strides

        # Default to ReLUs. Moreover, if `activations` is an activation function, repeat
        # it for every layer.
        activations = activations or nn.ReLU()
        if not isinstance(activations, (tuple, list)):
            activations = (activations,) * len(channels)
        elif len(activations) != len(channels):
            raise ValueError(
                f"Length of `activations` ({len(activations)}) must equal "
                f"the length of `channels` ({len(channels)})."
            )
        self.activations = activations

        # Compute number of halving layers.
        self.num_halving_layers = len(channels)

        # Compute receptive field at all stages of the model.
        self.receptive_fields = [1]
        # Forward pass:
        for stride, kernel in zip(self.strides, self.kernels):
            # Deal with composite kernels:
            if isinstance(kernel, tuple):
                kernel = kernel[0] + sum([k - 1 for k in kernel[1:]])
            after_conv = self.receptive_fields[-1] + (kernel - 1)
            if stride > 1:
                if after_conv % 2 == 0:
                    # If even, then subsample.
                    self.receptive_fields.append(after_conv // 2)
                else:
                    # If odd, then average pool.
                    self.receptive_fields.append((after_conv + 1) // 2)
            else:
                self.receptive_fields.append(after_conv)
        # Backward pass:
        for stride, kernel in zip(reversed(self.strides), reversed(self.kernels)):
            # Deal with composite kernels:
            if isinstance(kernel, tuple):
                kernel = kernel[0] + sum([k - 1 for k in kernel[1:]])
            if stride > 1:
                after_interp = self.receptive_fields[-1] * 2 - 1
                self.receptive_fields.append(after_interp + (kernel - 1))
            else:
                self.receptive_fields.append(self.receptive_fields[-1] + (kernel - 1))
        self.receptive_field = self.receptive_fields[-1]

        # If none of the fancy features are used, use the standard `nn.Conv` for
        # compatibility with trained models. For the same reason we also don't use the
        #   `activation` keyword.
        # TODO: In the future, use `self.nps.Conv` everywhere and use the `activation`
        #   keyword.
        if residual or separable or any(isinstance(k, tuple) for k in kernels):
            Conv = partial(
                self.nps.Conv,
                dim=dim,
                residual=residual,
                separable=separable,
            )
        else:

            def Conv(*, stride=1, transposed=False, kernel_size=None, **kw_args):
                padding = kernel_size // 2
                kw_args["kernel_size"] = kernel_size
                kw_args["padding"] = padding
                if transposed and stride > 1:
                    kw_args["output_padding"] = stride // 2
                if transposed:
                    return nn.ConvTranspose1d(stride=stride, **kw_args)
                else:
                    return nn.Conv1d(stride=stride, **kw_args)

        def construct_before_turn_layer(i):
            # Determine the configuration of the layer.
            ci = ((in_channels,) + tuple(channels))[i]
            co = channels[i]
            k = self.kernels[i]
            s = self.strides[i]

            if s == 1:
                # Just a regular convolutional layer.
                return Conv(
                    in_channels=ci,
                    out_channels=co,
                    kernel_size=k,
                    dtype=dtype,
                )
            else:
                # This is a downsampling layer.
                if self.receptive_fields[i] % 2 == 1:
                    # Perform average pooling if the previous receptive field is odd.
                    return nn.Sequential(
                        Conv(
                            in_channels=ci,
                            out_channels=co,
                            kernel_size=k,
                            stride=1,
                            dtype=dtype,
                        ),
                        nn.AvgPool1d(
                            kernel_size=s,
                            stride=s,
                        ),
                    )
                else:
                    # Perform subsampling if the previous receptive field is even.
                    return Conv(
                        in_channels=ci,
                        out_channels=co,
                        kernel_size=k,
                        stride=s,
                        dtype=dtype,
                    )

        def construct_after_turn_layer(i):
            # Determine the configuration of the layer.
            if i == len(channels) - 1:
                # No skip connection yet.
                ci = channels[i]
            else:
                # Add the skip connection.
                ci = 2 * channels[i]
            co = ((channels[0],) + tuple(channels))[i]
            k = self.kernels[i]
            s = self.strides[i]

            if s == 1:
                # Just a regular convolutional layer.
                return Conv(
                    in_channels=ci,
                    out_channels=co,
                    kernel_size=k,
                    dtype=dtype,
                )
            else:
                # This is an upsampling layer.
                if resize_convs:
                    return nn.Sequential(
                        nn.Upsample(scale_factor=s, mode=resize_conv_interp_method),
                        Conv(
                            in_channels=ci,
                            out_channels=co,
                            kernel_size=k,
                            stride=1,
                            dtype=dtype,
                        ),
                    )
                else:
                    return Conv(
                        in_channels=ci,
                        out_channels=co,
                        kernel_size=k,
                        stride=s,
                        transposed=True,
                        dtype=dtype,
                    )

        self.before_turn_layers = nn.ModuleList(
            [construct_before_turn_layer(i) for i in range(len(channels))]
        )
        self.after_turn_layers = nn.ModuleList(
            [construct_after_turn_layer(i) for i in range(len(channels))]
        )

        self.final_linear = nn.Conv1d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        x, uncompress = compress_batch_dimensions(x, self.dim + 1)

        hs = [self.activations[0](self.before_turn_layers[0](x))]
        for layer, activation in zip(
            self.before_turn_layers[1:],
            self.activations[1:],
        ):
            hs.append(activation(layer(hs[-1])))

        # Now make the turn!

        h = self.activations[-1](self.after_turn_layers[-1](hs[-1]))
        for h_prev, layer, activation in zip(
            reversed(hs[:-1]),
            reversed(self.after_turn_layers[:-1]),
            reversed(self.activations[:-1]),
        ):
            h = activation(layer(B.concat(h_prev, h, axis=1)))

        return uncompress(self.final_linear(h))
