#!/usr/bin/env python3

import sys
# FIXME(ycho): needed?
sys.path.append('../')
import unittest

import torch as th
from model import SceneEncoder


class TestOps(unittest.TestCase):
    def test_narrow(self):
        seed: int = 0
        num_iter: int = 128

        rng_state = th.get_rng_state()
        try:
            th.manual_seed(seed)
            for _ in range(num_iter):
                rank = th.randint(1, 9, ())
                shape = th.randint(1, 8, (rank,))
                dim = int(th.randint(0, rank, ()))
                x = th.randn(size=tuple(int(x) for x in shape))

                start = int(th.randint(0, shape[dim], ()))
                length = int(th.randint(0, shape[dim] - start, ()))
                x1 = th.narrow(x, dim, start, length)

                I = [slice(None) for _ in range(rank)]
                I[dim] = slice(start, start + length)
                x2 = x[I]
                assert(th.isclose(x1, x2).all())
        finally:
            th.set_rng_state(rng_state)


if __name__ == '__main__':
    unittest.main()
