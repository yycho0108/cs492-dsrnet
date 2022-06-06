#!/usr/bin/env python3
"""Visualization for the results from our model."""

from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Union, Optional
from tqdm.auto import tqdm
from model import DSRNet
import open3d as o3d
import einops
import numpy as np
from matplotlib import pyplot as plt

import cv2
import torch as th
nn = th.nn
F = nn.functional
from data_loader import DSRDataset
from torch.utils.data import DataLoader


@dataclass
class Config:
    data_path: str = '/media/ssd/datasets/DSR/real_test_data/'
    ckpt_file: str = '/tmp/host/dsr-019.pt'
    batch_size: int = 1
    num_frames: int = 10
    subseq_len: int = 10
    device: str = 'cuda'


def flow_to_image(
        flow: Union[np.ndarray, th.Tensor],
        max_flow: Optional[float] = None, eps: float = 1e-3) -> np.ndarray:
    flow = flow[:2]  # take x-y parts only
    if isinstance(flow, th.Tensor):
        flow = flow.detach().cpu().numpy()
    flow = np.asarray(flow, dtype=np.float32)
    x = flow[0]
    y = flow[1]

    # Project to 2D.
    # NOTE(ycho): somewhat hacky ...
    x = x.mean(axis=-1)
    y = y.mean(axis=-1)
    rho, theta = cv2.cartToPolar(x, y)

    if max_flow is None:
        max_flow = np.maximum(np.max(rho), eps)

    hsv = np.zeros(list(rho.shape) + [3], dtype=np.uint8)
    hsv[..., 0] = theta * 90 / np.pi
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(rho / max_flow, 1) * 255

    im = cv2.cvtColor(hsv, code=cv2.COLOR_HSV2RGB)
    return im


def show_volume(volume, max_num_class: int = 5,
                background_class: int = 4):
    """Show instance volume."""
    if isinstance(volume, th.Tensor):
        volume = volume.detach().cpu().numpy()
    print(np.min(volume), np.max(volume))
    # NOTE(ycho):
    # volume: (B, S1, S2, S3)
    S1, S2, S3 = volume.shape[-3:]

    cmap = plt.cm.get_cmap('Set1')
    label_colors = cmap(np.linspace(0, 1, max_num_class))

    # NOTE(ycho): using default voxel size from DSRNet source
    voxel_size: float = 0.004

    label = volume
    x, y, z = voxel_size * np.mgrid[:S1, :S2, :S3]  # shape=(3, S1, S2, S3)

    clouds = []
    for i in range(max_num_class):
        if i == background_class:
            continue
        mask = (label == i)

        # Create point cloud.
        points = np.stack([x[mask], y[mask], z[mask]], axis=-1).reshape(-1, 3)
        print(F'num points = {len(points)}')
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))

        # Assign colors.
        rgb = label_colors[i][..., :3]
        colors = einops.repeat(rgb, 'c -> n c', n=len(points))
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        clouds.append(cloud)

    o3d.visualization.draw_geometries(clouds)


def load_ckpt(ckpt_file: str, model: nn.Module):
    ckpt_file = Path(ckpt_file)
    save_dict = th.load(str(ckpt_file))
    model.load_state_dict(save_dict['model'])


def main():
    cfg = Config()

    # Load dataset.
    dataset = DSRDataset(cfg.data_path, 'train',
                         cfg.num_frames, cfg.subseq_len)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    # Load model.
    device = th.device(cfg.device)
    model = DSRNet(True, True)
    model = model.to(device)
    load_ckpt(cfg.ckpt_file, model)
    model.eval()

    data = iter(loader).next()

    prv_state = None
    for seq_id in range(cfg.num_frames):
        inputs = data[seq_id]
        #if prv_state is not None:
        #    inputs['prv_state'] = prv_state
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with th.no_grad():
            outputs = model(inputs)
        prv_state = outputs['state'].detach()

    if False:
        if False:
            # show model
            outputs = model(data)
            classes = outputs['logit'].argmax(dim=1).squeeze(0)
            show_volume(classes, background_class=4)
        elif False:
            # show mask
            classes = data['mask_3d'].squeeze(0)
            show_volume(classes, background_class=0)
        else:
            # Show TSDF volume from the dataset.
            data['tsdf']

    # flow = data['scene_flow_3d'].squeeze(0)
    with th.no_grad():
        flow = outputs['motion'].squeeze(0)
        # flow = inputs['scene_flow_3d'].squeeze(0)
    flow_image = flow_to_image(flow)
    cv2.imwrite('/tmp/flow.png', flow_image)

    color_image = inputs['color_image'].squeeze(0)
    color_image = color_image.detach().cpu().numpy()
    cv2.imwrite('/tmp/color.png', color_image)
    # cv2.imshow('flow', flow_image)
    # cv2.waitKey(0)


if __name__ == '__main__':
    main()
