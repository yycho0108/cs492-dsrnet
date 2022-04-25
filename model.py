import torch as th
from typing import Dict, Tuple, Union
nn = th.nn
F = nn.functional


class Conv3D(nn.Module):
    """General 3D Convolution block, with 3x3x3 kernel."""

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


class Conv2D(nn.Module):
    """General 3D Convolution block, with 3x3 kernel."""

    def __init__(self,
                 c_in: int,
                 c_out: int,
                 stride: Union[int, Tuple[int, int]],
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
        layers.append(nn.Conv2d(
            c_in, c_out, (3, 3),
            stride, padding=1, bias=False))
        if use_bn:
            layers.append(nn.BatchNorm2d(c_out))
        layers.append(nn.LeakyReLU(lrelu_eps, inplace=True))
        if use_up:
            layers.append(
                nn.Upsample(
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.layers(x)


class Conv3DResBlock(nn.Module):
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
        enc_layers.append(Conv3DResBlock(128, 128))
        enc_layers.append(Conv3DResBlock(128, 128))
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
            use_up = (i in (0, 2, 4, 6))
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


class MotionProjector(nn.Module):

    def __init__(self):
        self.grids: th.Tensor = None

    def forward(self, inputs: Dict[str, th.Tensor]) -> th.Tensor:
        mask = inputs['mask']

        # Lots of shape manipulation and normalization w.r.t `mask`
        mask_object = torch.narrow(mask, 1, 0, K - 1)
        sum_mask = torch.sum(mask_object, dim=(2, 3, 4))
        heatmap = torch.unsqueeze(mask_object, dim=2) * self.grids.to(device)
        pivot_vec = torch.sum(heatmap, dim=(
            3, 4, 5)) / torch.unsqueeze(sum_mask, dim=2)

        # Concatenate the background dimensions.
        # [Important] The last one is the background!
        trans_vec = torch.cat([trans_vec, self.zero_vec.expand(
            B, -1, -1).to(device)], dim=1).unsqueeze(-1)
        rot_mat = torch.cat(
            [rot_mat, self.eye_mat.expand(B, 1, -1, -1).to(device)],
            dim=1)
        pivot_vec = torch.cat([pivot_vec, self.zero_vec.expand(
            B, -1, -1).to(device)], dim=1).unsqueeze(-1)

        grids_flat = self.grids_flat.to(device)
        grids_after_flat = rot_mat @ (
            grids_flat - pivot_vec) + pivot_vec + trans_vec
        motion = (grids_after_flat - grids_flat).view([B, K, 3, S1, S2, S3])

        motion = torch.sum(motion * torch.unsqueeze(mask, 2), 1)


class MotionPredictor(nn.Module):
    """DSR-Net Motion Predictor.

    Predicts scene flow from scene representation and action embeddings.
    Assumes motion_type \\in SE(3), which results in a combination of:
    * mask_decoder = MaskDecoder(K)
    * transform_decoder = TransformDecoder(se3euler, K-1)
    * se3 = SE3(se3euler)

    and the output looks like:
    mask_feature = SceneEncoder(...)


    ...
    logit, mask = mask_decoder(mask_feature)

    transform_param = transform_decoder(mask_feature, input_action)
    trans_vec, rot_mat = se3(transform_param)
    """

    def __init__(self, use_action: bool):
        super().__init__()
        # NOTE(ycho): `use_action` is only `False`
        # while we don't have ActionEmbedding() implemented.
        self.use_action: bool = use_action
        self.num_objects: int = 5
        self.num_params: int = 6

        self.conv3d0 = Conv3D(16, 8, 2)
        self.conv3d1 = Conv3D(16, 16, 2)
        self.conv3d2 = Conv3D(32, 32, 2)

        self.conv3d3 = Conv3D(32, 16, 1, use_up=True)
        self.conv3d4 = Conv3D(16, 8, 1, use_up=True)
        self.conv3d5 = Conv3D(8, 8, 1, use_up=True)
        self.conv3d6 = nn.Conv3d(8, 3, kernel_size=3, padding=1)

        self.project = nn.Conv3d(128, 128, (4, 4, 2))

        # 8, 64, 64, 64, 64 (,8)
        act1_channels = (8, 64, 64, 64, 64)
        act1_layers = []
        for i, (c_in, c_out) in enumerate(act1_channels[:-1],
                                          act1_channels[1:]):
            stride = (2 if i == 0 else 1)
            act1_layers.append(Conv2D(c_in, c_out, stride))
        self.action1 = nn.Sequential(*act1_layers)
        self.action1e = Conv2D(64, 8)

        # 64, 128, 128, 128, 128 (,16)
        act2_channels = (64, 128, 128, 128, 128)
        act2_layers = []
        for i, (c_in, c_out) in enumerate(act2_channels[:-1],
                                          act2_channels[1:]):
            stride = (2 if i == 0 else 1)
            act2_layers.append(Conv2D(c_in, c_out, stride))
        self.action2 = nn.Sequential(*act2_layers)
        self.action2e = Conv2D(128, 16)

    def forward(self, inputs: Dict[str, th.Tensor]) -> th.Tensor:
        if not self.use_action:
            action = th.zeros_like(x, size=(batch_size, 8, 128, 128))
        else:
            action = inputs['action']

        # Action features at each dimensions
        action0 = action
        action1 = self.action1(action0)
        action2 = self.action2(action1)

        # Repeat action embeddings across +z dim.
        action0e = th.unsqueeze(action0e, -1)
        action0e = action0e.expand([-1, -1, -1, -1, 48])
        action1e = th.unsqueeze(self.action1e(action1e), -1)
        action1e = action1e.expand([-1, -1, -1, -1, 24])
        action2e = th.unsqueeze(self.actino2e(action2e), -1)
        action2e = action2e.expand([-1, -1, -1, -1, 12])

        # Mask Features
        x = inputs['feature']

        x = th.cat([x, action0], dim=1)
        x = self.conv3d0(x)
        dx0 = x # residual output

        x = th.cat([x, action1], dim=1)
        x = self.conv3d1(x)
        dx1 = x # residual output

        x = th.cat([x, action2], dim=1)
        x = self.conv3d2(x)

        x = self.conv3d3(x)
        x = self.conv3d4(x + dx1)
        x = self.conv3d5(x + dx0)
        x = self.conv3d6(x) # at this point, we have `motion_pred`

        x = self.project(x)  # -> should be Bx128x1x1x1
        x = x.squeeze(dim=(2, 3, 4))  # Bx128
        x = self.transform_mlp(x)  # Bx(NxP)
        x = x.reshape(-1, self.num_objects, self.num_params)  # BxNxP

        # NOTE(ycho): currently editing here
