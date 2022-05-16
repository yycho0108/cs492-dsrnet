#!/usr/bin/env python3

import sys
# FIXME(ycho): needed?
sys.path.append('../')
import unittest

import torch as th
from model import SceneEncoder, MotionPredictor


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
        assert(state.shape == (batch_size, 8, 128, 128, 48))

    def test_motion_predictor(self):
        rng_state = th.get_rng_state()
        try:
            seed: int = 0
            batch_size: int = 1
            use_action: bool = True
            num_objects: int = 5
            num_params: int = 6
            device: th.device = th.device('cpu')

            th.manual_seed(seed)

            motion_predictor = MotionPredictor(
                use_action, num_objects, num_params).to(
                device=device)

            dummy = dict(
                clf=th.randn(
                    size=(batch_size, num_objects, 128, 128, 48),
                    dtype=th.float32).to(device=device),
                action=th.zeros(
                    size=(batch_size, 8, 128, 128),
                    dtype=th.float32).to(device=device),
                feature=th.zeros(
                    size=(batch_size, 8, 128, 128, 48),
                    dtype=th.float32).to(device=device))
            motion = motion_predictor(dummy)
            # since clf is randn(), shouldn't really
            # result in NaN...
            assert((~th.isnan(motion)).all())
            # NOTE(ycho): shape-only test
            assert(motion.shape == (batch_size, 3, 128, 128, 48))
        finally:
            th.set_rng_state(rng_state)


if __name__ == '__main__':
    unittest.main()
