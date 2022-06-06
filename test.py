#!/usr/bin/env python3

from dataclasses import dataclass
import itertools
from simple_parsing import ArgumentParser
from tqdm.auto import tqdm
from os import PathLike
from typing import List, Optional, Tuple, Union, Any, Dict
from pathlib import Path

import numpy as np
import torch as th
nn = th.nn
F = nn.functional
from data_loader import DSRDataset
from torch.utils.data import DataLoader
from model import DSRNet



def apply_permutation(pairwise: np.array, permutation):
    """
    Applies permutation to batched pairwise array
    """
    B, M, N = pairwise.shape
    assert M == N
    assert N == len(permutation)
    return np.stack([pairwise[:, i, permutation[i]] for i in range(N)], axis=-1)

def calc_pairwise_inter_union(logit, mask_gt):
    """
    Args
    logit_list: [B, K, S1, S2, S3], softmax, K-1 is empty
    mask_gt_list: [B, S1, S2, S3], 0 is empty
    Returns
    inter: [B, K-1, K-1], voxels in the intersection of ground truth and prediction
    union: [B, K-1, K-1], voxels in the union of ground truth and prediction
    """
    B, K, S1, S2, S3 = logit.size()
    K -= 1
    inter = np.zeros([B, K, K])
    union = np.zeros([B, K, K])
    mask_pred_onehot = F.one_hot(th.argmax(logit, dim=1))[..., :-1]
    mask_gt_onehot = F.one_hot(mask_gt)[..., 1:]
    for b in range(B):
        for i in range(K):
            for j in range(K):
                # TODO find a better way for doing this
                occup_pred = mask_pred_onehot[b, :, :, :, i].to(dtype=th.bool)
                occup_gt = mask_gt_onehot[b, :, :, :, j].to(dtype=th.bool)
                inter[b, i, j] = th.sum(occup_gt * occup_pred).item()
                union[b, i, j] = th.sum(occup_gt + occup_pred).item()

    return inter, union

def calc_iou_for_permutation(inter, union, permutation):
    """
    inter: batched pairwise intersection
    union: batched pairwise union
    """
    inter_permuted = apply_permutation(inter, permutation)
    union_permuted = apply_permutation(union, permutation)
    # TODO how to handle empty union?
    return np.mean(inter_permuted / np.maximum(union_permuted, 1), axis=1)

def calc_iou_from_prediction(logit_list, mask_gt_list, ordered):
    B, K, _, _, _ = logit_list[0].size()
    L = len(logit_list)
    inter_union_list = [
        calc_pairwise_inter_union(logit, mask_gt)
        for logit, mask_gt in zip(logit_list, mask_gt_list)]
    if ordered:
        # Apply the same permutation for entire sequence, and get the best result
        best_iou = np.zeros(B)
        for permutation in itertools.permutations(range(K - 1)):
            frame_iou_sum = np.zeros(B)
            for inter, union in inter_union_list:
                frame_iou_sum += calc_iou_for_permutation(inter, union, permutation)
            best_iou = np.maximum(best_iou, frame_iou_sum / L)
        return best_iou
    else:
        # Try all permutations for each frame, and average the result over sequence
        frame_iou_sum = np.zeros(B)
        for inter, union in inter_union_list:
            best_iou = np.zeros(B)
            for permutation in itertools.permutations(range(K - 1)):
                best_iou = np.maximum(best_iou, calc_iou_for_permutation(inter, union, permutation))
            frame_iou_sum += best_iou
        return frame_iou_sum / L


@dataclass
class Config:
    data_path: str = '/media/ssd/datasets/DSR/real_test_data/'
    # The number of frames in a sequence
    num_frames: int = 10
    batch_size: int = 2
    # Use scene warping layer.
    use_warp: bool = True
    # Use action embeddings as inputs.
    use_action: bool = True
    device: str = 'cuda'
    load_ckpt_file: str = ''




def load_ckpt(ckpt_file: str, model: nn.Module):
    ckpt_file = Path(ckpt_file)
    save_dict = th.load(str(ckpt_file))
    model.load_state_dict(save_dict['model'])


def main():
    parser = ArgumentParser()
    parser.add_arguments(Config, dest='cfg')
    cfg = parser.parse_args().cfg

    device = th.device(cfg.device)
    dataset = DSRDataset(cfg.data_path, 'test', cfg.num_frames)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=1,
        prefetch_factor=4,
        pin_memory=True,
    )

    model = DSRNet(cfg.use_warp, cfg.use_action)
    model = model.to(device)

    load_ckpt(cfg.load_ckpt_file, model)

    # Run tests
    model.eval()
    # 0.4cm is the voxel size
    mse_scale_factor = 0.4 ** 2
    mse_surf_list = []
    mse_all_list = []
    ordered_iou_list = []
    unordered_iou_list = []
    for data in tqdm(loader, leave=False, desc='batch'):
        prv_state = None
        logit_list = []
        mask_gt_list = []
        for i in range(cfg.num_frames):
            inputs = data[i]
            if prv_state is not None:
                inputs['prv_state'] = prv_state
            outputs = model({k: v.to(device) for k, v in inputs.items()})
            prv_state = outputs['state'].detach()
            # .to(device) on inputs vs. cpu() on outputs
            tsdf = inputs['tsdf'].to(device)
            motion_pred = outputs['motion'].detach()
            motion_gt = inputs['scene_flow_3d'].to(device)
            logit = outputs['logit'].detach()
            mask_gt = inputs['mask_3d'].to(device)

            logit_list.append(logit)
            mask_gt_list.append(mask_gt)

            # Taken and adapted from the authors' code
            surface_mask = (tsdf > -0.99) * (tsdf < 0) # At surface?
            surface_mask = surface_mask * (mask_gt > 0) # Is object?
            surface_mask[..., 0] = 0 # Ignore background
            surface_mask = surface_mask.unsqueeze(1) # Add one dimension for XYZ

            squared_error = (motion_pred - motion_gt) ** 2
            mse_surf = th.sum(squared_error * surface_mask, dim=(1, 2, 3, 4)) / th.sum(surface_mask, dim=(1, 2, 3, 4))
            mse_all = th.mean(squared_error, dim=(1, 2, 3, 4))
            mse_surf_list.append(mse_surf)
            mse_all_list.append(mse_all)

        ordered_iou_list.append(calc_iou_from_prediction(logit_list, mask_gt_list, True))
        unordered_iou_list.append(calc_iou_from_prediction(logit_list, mask_gt_list, False))

    mse_surf = th.mean(th.cat(mse_surf_list)).item() * mse_scale_factor
    mse_all = th.mean(th.cat(mse_all_list)).item() * mse_scale_factor
    ordered_iou = np.mean(np.concatenate(ordered_iou_list))
    unordered_iou = np.mean(np.concatenate(unordered_iou_list))

    print(f'{mse_surf=}')
    print(f'{mse_all=}')
    print(f'{ordered_iou=}')
    print(f'{unordered_iou=}')

if __name__ == '__main__':
    main()
