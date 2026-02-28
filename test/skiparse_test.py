
import torch
from torchdiff.modules.osp_next import SkiparseRearrange, RearrangeType

test_num = 64
sparse_ratio = 4
samples = torch.arange(0, 64)
samples = samples.unsqueeze(0).unsqueeze(-1)

skiparse_1d_single = SkiparseRearrange(sparse_ratio, RearrangeType.Skiparse1DSingle)
skiparse_1d_single_reverse = SkiparseRearrange(sparse_ratio, RearrangeType.Skiparse1DSingleReverse)
skiparse_1d_group = SkiparseRearrange(sparse_ratio, RearrangeType.Skiparse1DGroup)
skiparse_1d_group_reverse = SkiparseRearrange(sparse_ratio, RearrangeType.Skiparse1DGroupReverse)
skiparse_1d_single_to_group = SkiparseRearrange(sparse_ratio, RearrangeType.Skiparse1DSingle2Group)
skiparse_1d_group_to_single = SkiparseRearrange(sparse_ratio, RearrangeType.Skiparse1DGroup2Single)

print(f"origin: \n: {samples}")

print(f"skiparse_1d_single: \n: {skiparse_1d_single(samples)}")

print(f"skiparse_1d_single_reverse: \n: {skiparse_1d_single_reverse(skiparse_1d_single(samples))}")

print(f"skiparse_1d_group: \n: {skiparse_1d_group(samples)}")

print(f"skiparse_1d_group_reverse: \n: {skiparse_1d_group_reverse(skiparse_1d_group(samples))}")

print(f"skiparse_1d_single_to_group: \n: {skiparse_1d_single_to_group(skiparse_1d_single(samples))}")

print(f"skiparse_1d_group_to_single: \n: {skiparse_1d_group_to_single(skiparse_1d_group(samples))}")

sparse_ratio = 2
skiparse_2d_single = SkiparseRearrange(sparse_ratio, RearrangeType.Skiparse2DSingle)
skiparse_2d_single_reverse = SkiparseRearrange(sparse_ratio, RearrangeType.Skiparse2DSingleReverse)
skiparse_2d_group = SkiparseRearrange(sparse_ratio, RearrangeType.Skiparse2DGroup)
skiparse_2d_group_reverse = SkiparseRearrange(sparse_ratio, RearrangeType.Skiparse2DGroupReverse)
skiparse_2d_single_to_group = SkiparseRearrange(sparse_ratio, RearrangeType.Skiparse2DSingle2Group)
skiparse_2d_group_to_single = SkiparseRearrange(sparse_ratio, RearrangeType.Skiparse2DGroup2Single)

print(f"origin: \n: {samples}")

print(f"skiparse_2d_single: \n: {skiparse_2d_single(samples, grid_sizes=(1, 8, 8))}")

print(f"skiparse_2d_single_reverse: \n: {skiparse_2d_single_reverse(skiparse_2d_single(samples, grid_sizes=(1, 8, 8)), grid_sizes=(1, 8, 8))}")

print(f"skiparse_2d_group: \n: {skiparse_2d_group(samples, grid_sizes=(1, 8, 8))}")

print(f"skiparse_2d_group_reverse: \n: {skiparse_2d_group_reverse(skiparse_2d_group(samples, grid_sizes=(1, 8, 8)), grid_sizes=(1, 8, 8))}")

print(f"skiparse_2d_single_to_group: \n: {skiparse_2d_single_to_group(skiparse_2d_single(samples, grid_sizes=(1, 8, 8)), grid_sizes=(1, 8, 8))}")

print(f"skiparse_2d_group_to_single: \n: {skiparse_2d_group_to_single(skiparse_2d_group(samples, grid_sizes=(1, 8, 8)), grid_sizes=(1, 8, 8))}")

