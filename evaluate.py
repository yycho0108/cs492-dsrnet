#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Iterable, Dict, Callable, List
from tqdm.auto import tqdm
from model import DSRNet
from pathlib import Path
# from torchvision.utils import flow_to_image

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


def flow_mse(
        inputs: Dict[str, th.Tensor],
        outputs: Dict[str, th.Tensor]) -> float:
    return F.mse_loss(outputs['motion'],
                      inputs['scene_flow_3d'])


def mask_iou(
        inputs: Dict[str, th.Tensor],
        outputs: Dict[str, th.Tensor]) -> float:
    # refer to PhysSong/cs492-dsrnet@iou-metrics
    pass


def evaluate_metric(cfg, model: nn.Module, loader: DataLoader, metric_fns: Dict[str, Callable[
        [Dict[str, th.Tensor], Dict[str, th.Tensor]], float]]) -> Dict[str, List[float]]:
    """Evaluate metrics through a dataset."""
    device = th.device(cfg.device)
    metrics = {k: [] for k in metric_fns.keys()}
    for data in tqdm(loader):
        prv_state = None
        for seq_id in range(cfg.num_frames):
            inputs = data[seq_id]
            #if prv_state is not None:
            #    inputs['prv_state'] = prv_state
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(inputs)
            prv_state = outputs['state'].detach()
            for k, f in metric_fns.items():
                metrics[k].append(f(inputs, outputs))
    return metrics


def show_flow(flow_image):
    """Show scene flow."""
    pass


def show_volume(volume):
    """Show instance volume."""
    pass


def load_ckpt(ckpt_file: str, model: nn.Module):
    ckpt_file = Path(ckpt_file)
    save_dict = th.load(str(ckpt_file))
    model.load_state_dict(save_dict['model'])


def main():
    cfg = Config()
    device = th.device(cfg.device)

    # Load dataset.
    dataset = DSRDataset(cfg.data_path, 'train',
                         cfg.num_frames, cfg.subseq_len)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    # Load model.
    model = DSRNet(True, True)
    model = model.to(device)
    load_ckpt(cfg.ckpt_file, model)
    model.eval()

    metric_fns = {'flow': flow_mse}
    evaluate_metric(cfg, model, loader,
                    metric_fns)


if __name__ == '__main__':
    main()
