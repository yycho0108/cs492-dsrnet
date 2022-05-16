#!/usr/bin/env python3

from dataclasses import dataclass
from simple_parsing import ArgumentParser
from tqdm.auto import tqdm

import torch as th
nn = th.nn
F = nn.functional
from data_loader import DSRDataset
from torch.utils.data import DataLoader
from model import DSRNet


@dataclass
class Config:
    data_path: str = '/media/ssd/datasets/DSR/real_test_data/'
    num_frames: int = 10
    batch_size: int = 2
    num_epoch: int = 30
    # Use scene warping layer.
    use_warp: bool = False
    # Use action embeddings as inputs.
    use_action: bool = False
    device: str = 'cuda'


def main():
    parser = ArgumentParser()
    parser.add_arguments(Config, dest='cfg')
    cfg = parser.parse_args().cfg

    device = th.device(cfg.device)
    dataset = DSRDataset(cfg.data_path, 'train', cfg.num_frames)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size
    )

    model = DSRNet(cfg.use_warp, cfg.use_action)
    model = model.to(device)

    optimizer = th.optim.Adam(model.parameters())
    motion_loss = nn.MSELoss()

    for epoch in tqdm(range(cfg.num_epoch), desc='epoch'):
        # lr = adjust_learning_rate(...)
        for data in tqdm(loader, leave=False, desc='batch'):
            # Formatted as TxBx[...]
            for i in range(cfg.num_frames):
                inputs = data[i]
                outputs = model({k: v.to(device) for k, v in inputs.items()})
                #inputs['motion'] = inputs['scene_flow_3d']
                inputs['prv_state'] = outputs['state'].detach()
                loss = motion_loss(outputs['motion'],
                                   inputs['scene_flow_3d'].to(device))
                print(F'loss = {loss.item()}')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


if __name__ == '__main__':
    main()
