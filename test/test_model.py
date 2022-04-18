#!/usr/bin/env python3

import sys
# FIXME(ycho): needed?
sys.path.append('../')
import unittest

import torch as th
from model import SceneEncoder


class TestSceneEncoder(unittest.TestCase):
    def test_scene_encoder(self):
        batch_size: int = 1
        device: th.device = th.device('cpu')

        scene_encoder = SceneEncoder().to(device=device)

        dummy = dict(
            tsdf=th.zeros(
                size=(batch_size, 128, 128, 48),
                dtype=th.float32).to(device=device),
            prv_state=th.zeros(
                size=(batch_size, 8, 128, 128, 48),
                dtype=th.float32).to(device=device))
        state = scene_encoder(dummy)

        # NOTE(ycho): shape-only test
        assert(state.shape == (batch_size, 8, 64, 64, 24))


if __name__ == '__main__':
    unittest.main()
