
import itertools
import torch as th
from typing import Dict, Tuple, Union, Iterable
nn = th.nn
F = nn.functional
import einops


def pairwise(seq: Iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(seq)
    next(b, None)
    return zip(a, b)


def epairwise(seq: Iterable):
    return enumerate(pairwise(seq))


class Conv3D3x3(nn.Module):
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


class Conv2D3x3(nn.Module):
    """General 2D Convolution block, with 3x3 kernel."""

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
        for i, (c_in, c_out) in epairwise(enc_channels):
            stride = (2 if (i in (0, 1, 5, 9)) else 1)
            enc_layers.append(Conv3D3x3(c_in, c_out, stride))
        enc_layers.append(Conv3DResBlock(128, 128))
        enc_layers.append(Conv3DResBlock(128, 128))
        split = enc_layers[:1], enc_layers[1:5], enc_layers[5:9], enc_layers[9:]
        self.enc = nn.ModuleList([nn.Sequential(*l) for l in split])

        # Decoder part
        dec_channels: Tuple[int, ...] = (128, 64, 64, 32, 32, 16, 16, 8, 8)
        residual_inputs = {2: 64, 4: 32, 6: 16}
        dec_layers: List[nn.Module] = []
        for i, (c_in, c_out) in epairwise(dec_channels):
            # Concatenate skip connections
            c_in += residual_inputs.get(i, 0)
            use_up = (i in (0, 2, 4, 6))
            dec_layers.append(Conv3D3x3(c_in, c_out, 1, use_up=use_up))
        split = dec_layers[:2], dec_layers[2:4], dec_layers[4:6], dec_layers[6:]
        self.dec = nn.ModuleList([nn.Sequential(*l) for l in split])

    def forward(self, inputs: Dict[str, th.Tensor]) -> th.Tensor:
        # FIXME(ycho): for some reason _always_
        # requires a batch dimension to be available,
        # even if B==1.
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
    """Project object-centric motion parameters to dense vector field."""

    def __init__(self, num_object: int):
        super().__init__()
        self.xyz: th.Tensor = th.stack(th.meshgrid(
            th.arange(128),
            th.arange(128),
            th.arange(48)),
            dim=0)
        self.num_object: int = num_object

    def forward(self, inputs: Dict[str, th.Tensor]) -> th.Tensor:
        """Predict the motion vector field.

        Args:
            inputs: Dictionary with the following entries:
                clf: Object occupancy probability, (B,K,W,H,D),
                    where K=number of object instances.
                txn: Translation vector, (B,K-1,3)
                    where K=number of object instances.
                rxn: Rotation parameters, (B,K-1,3)
                    where K=number of object instances.

        Returns:
            Motion vector prediction, (B,3,W,H,D).
                axis 1 represents coordinates in the order of (dx,dy,dz).
        """

        K: int = self.num_object
        clf: th.Tensor = inputs['clf']  # ..., K, [... volume ...]
        txn: th.Tensor = inputs['txn']  # ..., 3
        rxn: th.Tensor = inputs['rxn']  # ..., 3x3
        device = clf.device

        # Occupancy probability of each object instance.
        # NOTE(ycho): K-1, since the last channel is reserved for `background`.
        clf_object = th.narrow(clf, 1, 0, K - 1)  # clf[:, :K-1]
        voxel_count = einops.reduce(clf_object, 'b n ... -> b n', 'sum')
        # Sum of coordinates weighted by occupancy probabilities.
        numer = th.einsum(
            'b k ..., d ... -> b k d',
            clf_object,
            self.xyz.to(device, dtype=clf_object.dtype))
        denom = th.unsqueeze(voxel_count, dim=2)
        # NOTE(ycho):
        # `piv` is roughly the center of mass.
        piv = numer / denom

        # Add identity transforms at the end of the channel,
        # B (K-1) ... -> B K ...
        txn0 = th.zeros_like(txn[:, :1])
        txn = th.cat([txn, txn0], dim=1).unsqueeze(-1)
        rxn0 = einops.repeat(th.eye(3, dtype=rxn.dtype,
                                    device=rxn.device),
                             'i o -> b k i o', b=rxn.shape[0], k=1)
        rxn = th.cat([rxn, rxn0], dim=1)
        piv0 = th.zeros_like(piv[:, :1])
        piv = th.cat([piv, piv0], dim=1).unsqueeze(-1)

        xyz = einops.repeat(self.xyz.to(device, dtype=clf_object.dtype),
                            'd x y z -> b k d (x y z)',
                            b=rxn.shape[0], k=K)
        xyz2 = rxn @ (xyz - piv) + piv + txn
        motion = (xyz2 - xyz).view(*xyz.shape[:3], *self.xyz.shape[-3:])
        motion = th.sum(motion * th.unsqueeze(clf, 2), 1)
        return motion


class SO3(nn.Module):
    """Euler angle parameterization of the rotation matrix."""

    def __init__(self):
        super().__init__()

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Rotation matrix from euler angle parameterization.

        Args:
            x: Euler angles of shape (..., 3).
                (roll,pitch,yaw) convention.
        Returns:
            Rotation matrix of shape (..., 3, 3),
                R@x convention.
        """

        c, s = th.cos(x), th.sin(x)
        cx, cy, cz = [c[..., i] for i in range(3)]
        sx, sy, sz = [s[..., i] for i in range(3)]

        # NOTE(ycho):
        # Represents Rz@Ry@Rx matrix,
        # Which is equivalent to:
        # rotate_yaw(rotate_pitch(rotate_roll(point))).
        r = th.stack([
            cy * cz,
            (sx * sy * cz) - (cx * sz),
            (cx * sy * cz) + (sx * sz),
            cy * sz,
            (sx * sy * sz) + (cx * cz),
            (cx * sy * sz) - (sx * cz),
            -sy,
            (sx * cy),
            (cx * cy)], dim=-1)

        r = einops.rearrange(r, '... (i o) -> ... i o', i=3, o=3)
        return r


class MotionPredictor(nn.Module):
    """DSR-Net Motion Predictor.

    Predicts scene flow from scene representation and action embeddings.
    Uses euler angle parameterization internally.
    """

    def __init__(self, use_action: bool, num_objects: int = 5,
                 num_params: int = 6):
        """
        Args:
            use_action: Whether `action` embeddings will be provided as input.
                Otherwise, will generate a zero-filled tensor with same shape.
            num_objects: Maximum number of object instances to predict.
            num_params: The number of parameters for the transform encoding.
                By default, uses 6 = (3 for translation, 3 for rotation).
        """
        super().__init__()
        # NOTE(ycho): `use_action` is only `False`
        # while we don't have ActionEmbedding() implemented.
        self.use_action: bool = use_action
        self.num_objects: int = num_objects
        self.num_params: int = num_params

        self.conv3d0 = Conv3D3x3(8 + 8, 8, 2)
        self.conv3d1 = Conv3D3x3(8 + 8, 16, 2)
        self.conv3d2 = nn.Sequential(*[
            Conv3D3x3(i, o, s) for
            (i, o), s in zip(pairwise(
                [16 + 16, 32, 32, 32, 64]), (2, 1, 1, 1))])

        self.conv3d3 = nn.Sequential(
            Conv3D3x3(64, 128, 2),
            Conv3D3x3(128, 128, 2),
            nn.Conv3d(128, 128, kernel_size=(4, 4, 2))
        )

        # 8, 64, 64, 64, 64 (,8)
        act1_channels = (8, 64, 64, 64, 64)
        act1_layers = []
        for i, (c_in, c_out) in epairwise(act1_channels):
            stride = (2 if i == 0 else 1)
            act1_layers.append(Conv2D3x3(c_in, c_out, stride))
        self.action1 = nn.Sequential(*act1_layers)
        self.action1e = Conv2D3x3(64, 8, 1)

        # 64, 128, 128, 128, 128 (,16)
        act2_channels = (64, 128, 128, 128, 128)
        act2_layers = []
        for i, (c_in, c_out) in epairwise(act2_channels):
            stride = (2 if i == 0 else 1)
            act2_layers.append(Conv2D3x3(c_in, c_out, stride))
        self.action2 = nn.Sequential(*act2_layers)
        self.action2e = Conv2D3x3(128, 16, 1)

        # MLP
        # NOTE(ycho): last "object" is reserved for background.
        # WARN(ycho): 5x512 in the paper, 4x512 hidden layers in the author's
        # code.
        mlp_channels = (128, 512, 512, 512, 512,
                        self.num_params * (self.num_objects - 1))
        mlp_layers = []
        for i, (c_in, c_out) in epairwise(mlp_channels):
            use_lrelu = (i != 5)
            mlp_layers.append(nn.Linear(c_in, c_out))
            if use_lrelu:
                mlp_layers.append(nn.LeakyReLU(inplace=True))
        self.mlp = nn.Sequential(*mlp_layers)

        self.so3 = SO3()
        self.motion_projector = MotionProjector(self.num_objects)

    def forward(self, inputs: Dict[str, th.Tensor]) -> th.Tensor:
        """Predict the motion vector field.

        Args:
            inputs: Dictionary with the following entries:
                clf:     Object occupancy probability, (B,K,W,H,D),
                    where K=number of object instances.
                feature: Feature embedding, (B,S,W,H,D),
                    where S=128 state representation dimensionality.
                action:  Action encoding, (B,A,W,H,D),
                    where A=8 discrete action bins.

        Returns:
            Motion vector prediction, (B,3,W,H,D).
                axis 1 represents coordinates in the order of (dx,dy,dz).
        """
        if not self.use_action:
            batch_size: int = inputs['feature'].shape[0]
            action: th.Tensor = th.zeros(
                size=(batch_size, 8, 128, 128),
                dtype=th.float32,
                device=inputs['feature'].device)
        else:
            action: th.Tensor = inputs['action']
        clf: th.Tensor = inputs['clf']

        # Action features at each dimensions
        action0 = action
        action1 = self.action1(action0)
        action2 = self.action2(action1)

        # Repeat action embeddings across +z dim.
        action0e = einops.repeat(action0, '... -> ... c', c=48)
        action1e = einops.repeat(self.action1e(action1), '... -> ... c', c=24)
        action2e = einops.repeat(self.action2e(action2), '... -> ... c', c=12)

        # Mask Features
        x = inputs['feature']

        x = th.cat([x, action0e], dim=1)
        x = self.conv3d0(x)

        x = th.cat([x, action1e], dim=1)
        x = self.conv3d1(x)

        x = th.cat([x, action2e], dim=1)
        x = self.conv3d2(x)
        x = self.conv3d3(x)  # -> should be Bx128x1x1x1
        x = x.view(-1, 128)
        x = self.mlp(x)  # Bx(NxP)

        # NOTE(ycho): last "object" is reserved for background!
        x = x.view(-1, self.num_objects - 1,
                   self.num_params)  # B,K-1,P

        params = x
        rxn = self.so3(params[..., :3])  # B,K-1,3,3
        txn = params[..., 3:]  # B,K-1,3

        motion = self.motion_projector(dict(
            clf=clf,
            rxn=rxn,
            txn=txn
        ))
        return motion


class MaskPredictor(nn.Module):
    """TODO(physsong)"""

    def __init__(self, num_object: int):
        super().__init__()
        self.num_object = num_object
        self.layer = nn.Conv3d(8, num_object, kernel_size=1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.layer(x)


class DSRNet(nn.Module):
    def __init__(self, use_warp: bool = False, use_action: bool = False,
                 num_object: int = 5):
        super().__init__()
        # tsdf, prv_state -> cur_state
        self.scene_encoder = SceneEncoder()
        self.mask_predictor = MaskPredictor(num_object)
        # clf, feature [, action] -> motion
        # TODO(ycho): set `use_action` to True
        self.motion_predictor = MotionPredictor(use_action, num_object)
        # self.warp = SceneFlowWarper()
        self.use_warp = use_warp

    def forward(self, inputs: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        # NOTE(ycho): initialize
        # `prv_state` to all-zero tensor if not given.
        if 'prv_state' not in inputs:
            batch_size: int = inputs['tsdf'].shape[0]
            inputs['prv_state'] = th.zeros(
                size=[batch_size, 8, 128, 128, 48],
                dtype=th.float32,
                device=inputs['tsdf'].device)

        state = self.scene_encoder(inputs)
        logit = self.mask_predictor(state)
        clf = F.softmax(logit, dim=1)  # TODO(ycho): temperature?

        mp_inputs = dict(clf=clf, feature=state
                         #,action=None
                         )
        motion = self.motion_predictor(mp_inputs)

        outputs: Dict[str, th.Tensor] = {}
        outputs['logit'] = logit
        outputs['motion'] = motion
        if not self.use_warp:
            outputs['state'] = state
        elif 'motion' in inputs:
            warp_mask: th.Tensor = None
            outputs['state'] = self.warp(state,
                                         inputs['motion'], warp_mask)
        else:
            warp_mask: th.Tensor = None
            outputs['state'] = self.warp(state,
                                         motion, warp_mask)
        return outputs
