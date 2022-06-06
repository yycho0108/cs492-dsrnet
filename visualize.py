#!/usr/bin/env python3

from typing import Iterable
from tqdm.auto import tqdm
from model import DSRNet
from torchvision.utils import flow_to_image
import open3d as o3d
import einops
from matplotlib import pyplot as plt

import torch as th
nn = th.nn
F = nn.functional


@dataclass
class Config:
    ckpt_file: str = '/tmp/dsr-019.pt'
    batch_size: int = 1
    num_frames: int = 10


def show_flow(flow_image):
    """Show scene flow."""
    pass


def show_volume(volume, max_num_class: int = 5):
    """Show instance volume."""
    # NOTE(ycho):
    # volume: (B, S1, S2, S3)

    cmap = plt.cm.get_cmap('Set1')
    label_colors = cmap(np.linspace(0, 1, max_num_class))

    # NOTE(ycho): using default voxel size from DSRNet source
    voxel_size: float = 0.004

    label = volume
    x, y, z = voxel_size * np.mgrid[:S1, :S2, :S3]  # shape=(3, S1, S2, S3)

    clouds = []
    for i in range(max_num_class):
        mask = (label == i)

        # Create point cloud.
        points = np.stack([x[mask], y[mask], z[mask]], axis=-1).reshape(-1, 3)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))

        # Assign colors.
        rgb = label_colors[i][..., :3]
        colors = einops.repeat(rgb, 'c -> n c', n=len(points))
        cloud.colors = o3d.Vector3dVector(colors.astype(np.float64))

        clouds.append(cloud)

    o3d.visualization.draw_geometries(clouds)


def load_ckpt(ckpt_file: str, model: nn.Module):
    ckpt_file = Path(ckpt_file)
    save_dict = th.load(str(ckpt_file))
    model.load_state_dict(save_dict['model'])


def main():
    cfg = Config()

    # Load dataset.
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    # Load model.
    model = DSRNet(True, True)
    load_ckpt(ckpt_file, model)
    model.eval()

    metric_fns = {'flow': flow_mse}
    evaluate_metric(cfg, model, loader,
                    metric_fns)


if __name__ == '__main__':
    main()
