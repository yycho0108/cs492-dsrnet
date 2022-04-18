#!/usr/bin/env python3

import torch as th
from typing import Dict, Tuple, Union
nn = th.nn
F = nn.functional


class Conv3D(nn.Module):
    """General 3D Convolution block."""

    def __init__(self,
                 c_in: int,
                 c_out: int,
                 stride: Union[int, Tuple[int, int, int]],
                 use_bn: bool = True,
                 use_up: bool = False,
                 lrelu_eps: float = 0.2,
                 ):
        """

        Args:
            c_in: Number of input channels.
            c_out: Number of output channels.
            stride: Convolution stride.
            use_bn: Whether to use batch normalization. (nn.BatchNorm3d)
            use_up: Whether to use trilinear upsampling (nn.Upsample)
            lrelu_eps: nn.LeakyRelu slope parameter.
                WARN(ycho): defaults to 0.2 (according to the paper),
                but beware that the author's implementation seems to use 1e-2.
        """
        super().__init__()

        layers: List[nn.Module] = []
        layers.append(nn.Conv3d(
            c_in, c_out, (3, 3, 3),
            stride, padding=1, bias=False))
        if use_bn:
            layers.append(nn.BatchNorm3d(c_out))
        layers.append(nn.LeakyReLU(lrelu_eps, inplace=True))
        if use_up:
            layers.append(
                nn.Upsample(
                    scale_factor=2,
                    mode='trilinear',
                    align_corners=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.layers(x)


class ConvResBlock(nn.Module):
    """Convolutional residual block, x'=x+f(x)."""

    def __init__(self, c_in: int, c_out: int,
                 lrelu_eps: float = 1e-2):
        """

        Args:
            c_in: Number of input channels.
            c_out: Number of output channels.
            lrelu_eps: nn.LeakyRelu slope parameter.
        """
        super().__init__()
        layers: List[nn.Module] = []
        layers.append(nn.Conv3d(c_in, c_out, 3, padding=1, bias=False))
        layers.append(nn.BatchNorm3d(c_out))
        layers.append(nn.LeakyReLU(lrelu_eps, inplace=True))
        layers.append(nn.Conv3d(c_out, c_out, 3, padding=1, bias=False))
        layers.append(nn.BatchNorm3d(c_out))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return x + self.layers(x)


class SceneEncoder(nn.Module):
    """DSR-Net Scene Encoder.

    Produces current latent-state vector from TSDF-formatted
    observations and the previous state.
    """

    def __init__(self):
        super().__init__()
        # TODO(ycho): cleanup ugly syntax

        # Encoder part
        # NOTE(ycho): in author's implementation, number of input channels
        # are `12` (8+1+3) due to undocumented inclusion of
        # meshgrid "coordinate features" (const) rather than `9` (8+1).
        enc_channels: Tuple[int, ...] = (
            9,
            16, 32, 32, 32,
            32, 64, 64, 64, 64,
            128
        )
        enc_layers: List[nn.Module] = []
        for i, (c_in, c_out) in enumerate(zip(
                enc_channels[: -1],
                enc_channels[1:])):
            stride = (2 if (i in (0, 1, 5, 9)) else 1)
            enc_layers.append(Conv3D(c_in, c_out, stride))
        enc_layers.append(ConvResBlock(128, 128))
        enc_layers.append(ConvResBlock(128, 128))
        split = enc_layers[:1], enc_layers[1:5], enc_layers[5:9], enc_layers[9:]
        self.enc = [nn.Sequential(*l) for l in split]

        # Decoder part
        dec_channels: Tuple[int, ...] = (128, 64, 64, 32, 32, 16, 16, 8, 8)
        residual_inputs = {2: 64, 4: 32, 6: 16}
        dec_layers: List[nn.Module] = []
        for i, (c_in, c_out) in enumerate(
                zip(dec_channels[:-1], dec_channels[1:])):
            # Concatenate skip connections
            c_in += residual_inputs.get(i, 0)
            use_up = (i in (0, 2, 4, 8))
            dec_layers.append(Conv3D(c_in, c_out, 1, use_up=use_up))
        split = dec_layers[:2], dec_layers[2:4], dec_layers[4:6], dec_layers[6:]
        self.dec = [nn.Sequential(*l) for l in split]

    def forward(self, inputs: Dict[str, th.Tensor]) -> th.Tensor:
        cur_obs = inputs['tsdf'][:, None]  # Bx1x128x128x48
        prv_state = inputs['prv_state']  # Bx8x128x128x48

        # NOTE(ycho): assumes channel axis=1
        x = th.cat((prv_state, cur_obs), dim=1)

        # Encoder blocks: 0:1, 1:5, 5:9, 9:12(x)
        xs = []
        for f in self.enc:
            x = f(x)
            xs.append(x)

        # Decoder blocks: 0:3(x), 3:5, 5:7, 7:
        for i, f in enumerate(self.dec):
            if i > 0:
                x = th.cat((x, xs[-i - 1]), dim=1)
            x = f(x)
        return x
