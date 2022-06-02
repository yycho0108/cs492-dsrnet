#!/usr/bin/env python3

from dataclasses import dataclass
import itertools
from simple_parsing import ArgumentParser
from tqdm.auto import tqdm
from os import PathLike
from typing import List, Optional, Tuple, Union, Any, Dict
from pathlib import Path

import torch as th
nn = th.nn
F = nn.functional
from data_loader import DSRDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import DSRNet


@dataclass
class Config:
    data_path: str = '/media/ssd/datasets/DSR/real_test_data/'
    # The number of frames in a sequence
    num_frames: int = 10
    # The length of subsequences
    subseq_len: int = 10
    batch_size: int = 2
    num_epoch: int = 30
    # Use scene warping layer.
    use_warp: bool = False
    # Use action embeddings as inputs.
    use_action: bool = False
    device: str = 'cuda'
    # Weight for motion loss
    loss_motion_weight: float = 1.0
    # Weight for mask loss
    loss_mask_weight: float = 5.0
    load_ckpt_file: Optional[str] = None
    path: str = '/tmp/dsr'


def _ensure_dir(path: Union[str, PathLike]) -> Path:
    """ensure that directory exists."""
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def mask_loss(
    logit: th.Tensor,
    mask_gt: th.Tensor,
    mask_order: Optional[List[Tuple[int, ...]]] = None
) -> Tuple[th.Tensor, List[Tuple[int, ...]]]:
    """
    Get the mask loss for given mask_order and get the best order for the prediction
    if mask_order is not given, the best permutation for each batch will be used
    Arguments:
        logit: [B, K, S1, S2, S3], the mask predictor output logit(K-1 is none)
        mask_gt: [B, S1, S2, S3], the ground truth mask index(0 is none)
        mask_order: mask_order[b] is the permutation tuple of object in batch b(length K-1)
    """
    def loss_for_permutation(
        logit: th.Tensor,
        mask_gt: th.Tensor,
        permutation: Tuple[int, ...]
    ) -> th.Tensor:
        # CrossEntropyLoss on logits is equivalent to NLL on softmax result
        logit_permuted = th.stack(
            [logit[b:b + 1, -1]] + [logit[b:b + 1, i] for i in permutation], dim=1)
        return F.cross_entropy(logit_permuted, mask_gt[b:b + 1])
    loss = 0
    B, K, S1, S2, S3 = logit.size()
    best_mask_order = []
    for b in range(B):
        best_loss, best_perm = None, None
        input_permutation = mask_order[b] if mask_order is not None else None
        for permutation in itertools.permutations(range(K - 1)):
            cur_loss = loss_for_permutation(logit, mask_gt, permutation)
            # If mask_order is given, use that
            if permutation == input_permutation:
                loss += cur_loss
            if best_loss is None or cur_loss.item() < best_loss.item():
                best_loss = cur_loss
                best_perm = permutation
        best_mask_order.append(best_perm)
        # If mask_order is not provided, use the best one
        if input_permutation is None:
            loss += best_loss
    return loss, best_mask_order


def save_ckpt(ckpt_file: str, model: nn.Module,
              optimizer: Optional[th.optim.Optimizer] = None):
    ckpt_file = Path(ckpt_file)
    _ensure_dir(ckpt_file.parent)
    save_dict: Dict[str, Any] = {}
    save_dict['model'] = model.state_dict()
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    th.save(save_dict, str(ckpt_file))


def load_ckpt(ckpt_file: str, model: nn.Module,
              optimizer: Optional[th.optim.Optimizer] = None):
    ckpt_file = Path(ckpt_file)
    save_dict = th.load(str(ckpt_file))
    model.load_state_dict(save_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(save_dict['model'])


def add_tensorboard_graph(model: nn.Module, loader: DataLoader,
                          writer: SummaryWriter,
                          device: th.device):
    try:
        train_mode = model.training
        model.train(False)
        data = iter(loader).next()[0]
        keys = sorted(list(data.keys()))

        # NOTE(ycho): We need to do this workaround
        # because in torch < 1.10, the tracer
        # fails to track `dict`-valued inputs and outputs.
        class DummyModel(nn.Module):
            def __init__(self, model: nn.Module):
                super().__init__()
                self.model = model

            def forward(self, *args):
                return tuple(self.model(
                    {k: v.to(device) for k, v in zip(keys, args)}).values())
        writer.add_graph(DummyModel(model),
                         tuple(data[k].detach() for k in keys))
        writer.flush()
        # In case this op leaves any artifacts...
        th.cuda.empty_cache()
    finally:
        model.train(train_mode)


def main():
    parser = ArgumentParser()
    parser.add_arguments(Config, dest='cfg')
    cfg = parser.parse_args().cfg
    root = Path(cfg.path)
    index = len([d for d in root.glob('run-*') if d.is_dir()])
    path = _ensure_dir(Path(cfg.path) / F'run-{index:03d}')
    log_path = _ensure_dir(path / 'log')
    print(F'Runtime path = {path}')

    device = th.device(cfg.device)
    dataset = DSRDataset(cfg.data_path, 'train', cfg.num_frames, cfg.subseq_len)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    model = DSRNet(cfg.use_warp, cfg.use_action)
    model = model.to(device)

    optimizer = th.optim.Adam(model.parameters())
    motion_loss = nn.MSELoss()
    writer = SummaryWriter(log_path)

    # Add tensorboard graph visualization
    # by converting inputs/outputs to a dummy model.
    add_tensorboard_graph(model, loader, writer, device)
    loss_motion_weight = cfg.loss_motion_weight
    loss_mask_weight = cfg.loss_mask_weight

    def _train_step(pbar, data, step):
        # Formatted as TxBx[...]
        prv_state = None
        mask_order = None
        for i in range(cfg.subseq_len):
            inputs = data[i]
            if prv_state is not None:
                inputs['prv_state'] = prv_state
            outputs = model({k: v.to(device) for k, v in inputs.items()})
            prv_state = outputs['state'].detach()
            loss_motion = motion_loss(outputs['motion'],
                                      inputs['scene_flow_3d'].to(device))
            loss_mask, mask_order = mask_loss(
                outputs['logit'],
                inputs['mask_3d'].to(device),
                mask_order)
            loss = loss_motion_weight * loss_motion + loss_mask_weight * loss_mask
            writer.add_scalar('loss', loss.item(), step + i)
            pbar.set_postfix_str(F'loss={loss.item():03f}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return step + cfg.subseq_len

    # Optionally load checkpoint.
    if cfg.load_ckpt_file is not None:
        load_ckpt(cfg.load_ckpt_file, model, optimizer)

    # Train.
    try:
        step: int = 0
        model.train(True)
        for epoch in tqdm(range(cfg.num_epoch), desc='epoch'):
            # lr = adjust_learning_rate(...)
            with tqdm(loader, leave=False, desc='batch') as pbar:
                for data in pbar:
                    step = _train_step(pbar, data, step)
            save_ckpt(F'{path}/dsr-{epoch:03d}.pt', model, optimizer)
    finally:
        save_ckpt(F'{path}/dsr-last.pt', model, optimizer)


if __name__ == '__main__':
    main()
