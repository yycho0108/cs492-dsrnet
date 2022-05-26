from collections import namedtuple
from cupy import RawKernel
import numpy as np
import torch as th
from torch.autograd.function import Function

Stream = namedtuple('Stream', ['ptr'])
def get_current_cuda_stream():
    return Stream(ptr=th.cuda.current_stream().cuda_stream)

def scatter_interpolate_cuda(feature, new_coord):
    kernel = r'''
    extern "C"
    __global__ void scatter_interpolate(
        const float * __restrict__ feature_in, // [B, C, W, H, D]
        const float * __restrict__ new_coord, // [B, 3, W, H, D]
        float * __restrict__ feature_out, // [B, C, W, H, D]
        const int batch_size,
        const int dim_feature,
        const int dim_x,
        const int dim_y,
        const int dim_z
    )
    {
        int voxel_count = dim_x * dim_y * dim_z;

        int block_idx = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
        int thread_idx = threadIdx.x + blockDim.x * (threadIdx.y + block_idx * blockDim.y);

        int batch_index = thread_idx / voxel_count;
        int voxel_idx = thread_idx % voxel_count;
        if (batch_index >= batch_size) return;

        int feature_base = batch_index * dim_feature * voxel_count;
        int coord_base = batch_index * 3 * voxel_count;

        int voxel_idx_feature = feature_base + voxel_idx;
        int voxel_idx_coord = coord_base + voxel_idx;

        float x_float = new_coord[voxel_idx_coord];
        float y_float = new_coord[voxel_idx_coord + voxel_count];
        float z_float = new_coord[voxel_idx_coord + voxel_count + voxel_count];

        // NOTE: the author used truncation, which gives wrong result around -2<x,y,z<0
        int x_floor = floorf(x_float);
        int y_floor = floorf(y_float);
        int z_floor = floorf(z_float);

        for(int t = 0; t < 8; ++t) {
            int dx = (t >> 2) & 1;
            int dy = (t >> 1) & 1;
            int dz = (t >> 0) & 1;

            int x = x_floor + dx;
            int y = y_floor + dy;
            int z = z_floor + dz;
            float weight = fabsf(x_float - x) * fabsf(y_float - y) * fabsf(z_float - z);

            if (x >= 0 && x < dim_x && y >= 0 && y < dim_y && z >= 0 && z < dim_z) {
                int new_idx = (x * dim_y + y) * dim_z + z;
                int new_idx_feature = feature_base + new_idx;

                for(int c = 0; c < dim_feature; ++c) {
                    int offset = c * voxel_count;
                    atomicAdd(&feature_out[new_idx_feature + offset], feature_in[voxel_idx_feature + offset] * weight);
                }
            }
        }
    }
    '''
    f = RawKernel(kernel, 'scatter_interpolate')

    B, C, W, H, D = feature.size()

    threads_per_block = 1024
    n_blocks = int(np.ceil(B * W * H * D / threads_per_block))

    feature_contig = feature.contiguous()
    new_coord_contig = new_coord.contiguous()

    feature_new = th.zeros_like(feature)

    f(grid=(n_blocks, 1, 1),
        block=(threads_per_block, 1, 1),
        args=[
            feature_contig.data_ptr(),
            new_coord_contig.data_ptr(),
            feature_new.data_ptr(),
            B, C, W, H, D],
        stream=get_current_cuda_stream())

    return feature_new


class ScatterInterpolateCUDA(Function):
    @staticmethod
    def forward(ctx, feature, new_coord):
        feature_new = scatter_interpolate_cuda(feature, new_coord)
        return feature_new

    @staticmethod
    @once_differentiable
    def backward(ctx, feature_new_grad):
        # NOTE: this wasn't implemented in author's code
        # and implementing this doesn't seem really helpful
        return None, None

def scatter_interpolate(feature, new_coord):
    return ScatterInterpolateCUDA.apply(feature, new_coord)


def forward_warp(feature, flow, mask):
    B, C, W, H, D = feature.size()
    mask = mask.unsqueeze(1)
    orig_coord = th.stack(th.meshgrid(th.arange(W), th.arange(H), th.arange(D)), dim=0)
    new_coord = flow + orig_coord.to(flow.device)
    masked_feature = feature * mask

    feature_new = scatter_interpolate(masked_feature, new_coord)
    cnt = scatter_interpolate(mask, new_coord)

    eps=1e-3
    cnt = th.max(cnt, other=th.ones_like(cnt) * eps)
    feature_new = feature_new / cnt

    return feature_new
