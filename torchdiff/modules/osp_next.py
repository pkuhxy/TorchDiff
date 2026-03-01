# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange, repeat, reduce
from abc import ABC, abstractmethod

from torchdiff.distributed.cp_state import cp_state, use_context_parallel, use_skiparse_context_parallel
from torchdiff.distributed.communication import all_gather, all_to_all_4D, get_shard_seq_lens
from torchdiff.utils.utils import is_npu_available, safe_get_rank, contiguous, SafeCacheManager

from .attention import flash_attention, attention, attention_with_mask
from .want2v import (
    sinusoidal_embedding_1d,
    rope_params,
    WanLayerNorm as OSPNextLayerNorm,
    WanRMSNorm as OSPNextRMSNorm,
    Head,
)

T5_CONTEXT_TOKEN_NUMBER = 512

class SkiparseModelType:
    DualEnd = "dual_end"
    Uniform = "uniform"
    Full = "full"

class SkiparseBlockType:
    Full = "full"
    Single = "single"
    Group = "group"

class RearrangeType:

    # identity: 不需要rearrange
    Identity = "identity"
    # repeat: 在batch维度重复，用于text和register tokens
    Repeat = "repeat"
    # reduce: 在batch维度平均，用于register tokens
    Reduce = "reduce"

    Skiparse1DSingle = "skiparse_1d_single"
    Skiparse1DSingleReverse = "skiparse_1d_single_reverse"
    Skiparse1DGroup = "skiparse_1d_group"
    Skiparse1DGroupReverse = "skiparse_1d_group_reverse"
    Skiparse1DSingle2Group = "skiparse_1d_single_to_group"
    Skiparse1DGroup2Single = "skiparse_1d_group_to_single"

    Skiparse2DSingle = "skiparse_2d_single"
    Skiparse2DSingleReverse = "skiparse_2d_single_reverse"
    Skiparse2DGroup = "skiparse_2d_group"
    Skiparse2DGroupReverse = "skiparse_2d_group_reverse"
    Skiparse2DSingle2Group = "skiparse_2d_single_to_group"
    Skiparse2DGroup2Single = "skiparse_2d_group_to_single"

    @classmethod
    def input_is_full(cls, rearrange_type):
        return rearrange_type in [
            cls.Skiparse1DSingle,
            cls.Skiparse1DGroup,
            cls.Skiparse2DSingle,
            cls.Skiparse2DGroup,
        ]

    @classmethod
    def input_is_skiparse_1d(cls, rearrange_type):
        return rearrange_type in [
            cls.Skiparse1DSingleReverse,
            cls.Skiparse1DGroupReverse,
            cls.Skiparse1DSingle2Group,
            cls.Skiparse1DGroup2Single,
        ]

    @classmethod
    def input_is_skiparse_2d(cls, rearrange_type):
        return rearrange_type in [
            cls.Skiparse2DSingleReverse,
            cls.Skiparse2DGroupReverse,
            cls.Skiparse2DSingle2Group,
            cls.Skiparse2DGroup2Single,
        ]
    
    @classmethod
    def output_is_full(cls, rearrange_type):
        return rearrange_type in [
            cls.Skiparse1DSingleReverse,
            cls.Skiparse1DGroupReverse,
            cls.Skiparse2DSingleReverse,
            cls.Skiparse2DGroupReverse,
        ]

    @classmethod
    def output_is_skiparse_1d(cls, rearrange_type):
        return rearrange_type in [
            cls.Skiparse1DSingle,
            cls.Skiparse1DGroup,
            cls.Skiparse1DSingle2Group,
            cls.Skiparse1DGroup2Single,
        ]

    @classmethod
    def output_is_skiparse_2d(cls, rearrange_type):
        return rearrange_type in [
            cls.Skiparse2DSingle,
            cls.Skiparse2DGroup,
            cls.Skiparse2DSingle2Group,
            cls.Skiparse2DGroup2Single,
        ]

    @classmethod
    def is_single2group(cls, rearrange_type):
        return rearrange_type in [
            cls.Skiparse1DSingle2Group,
            cls.Skiparse2DSingle2Group,
        ]
    
    @classmethod
    def is_group2single(cls, rearrange_type):
        return rearrange_type in [
            cls.Skiparse1DGroup2Single,
            cls.Skiparse2DGroup2Single,
        ]

class SkiparseRearrange:
    def __init__(self, sparse_ratio=1, rearrange_type=RearrangeType.Identity):

        self.sparse_ratio = sparse_ratio
        self.rearrange_type = rearrange_type

        self.skiparse_1d = "skiparse_1d" in self.rearrange_type
        self.skiparse_2d = "skiparse_2d" in self.rearrange_type

        if self.skiparse_1d and self.skiparse_2d:
            raise ValueError(f"We can only use skiparse 1d or skiparse 2d, not both at the same time!")

        rearrange_func = f"_{rearrange_type}"
        if not hasattr(self, rearrange_func):
            raise ValueError(f"Unsupported rearrange operation: {rearrange_func}")
        self.rearrange_func = getattr(self, rearrange_func)

        # 缓存padding tokens的个数
        self.padding_cache = SafeCacheManager(max_cache_size=32)

    # ----------------- skiparse context parallel -----------------
    """
    skiparse context parallel:
        skiparse rearrange会将序列rearrange到batch维度，我们可以将batch维度划分到不同的skiparse context parallel processes，每个process处理一部分序列。
    """
    def _skiparse_cp_scatter(self, x, dim=0):
        if not use_skiparse_context_parallel():
            return x
        size = x.shape[dim]
        cp_size = cp_state.skiparse_cp_size
        assert size % cp_size == 0
        chunk_size = size // cp_size
        return x.narrow(dim, cp_state.skiparse_cp_rank * chunk_size, chunk_size)
    
    def _skiparse_cp_gather(self, x, dim=0):
        if not use_skiparse_context_parallel():
            return x
        x = all_gather(x, dim=dim, group=cp_state.skiparse_cp_group)
        return x

    def _identity(self, x, grid_sizes=None):
        return x

    # ================== einops实现 ==================
    # def _repeat(self, x, grid_sizes=None):
    #     x = repeat(x, 'b n c -> (b p) n c', p=self.sparse_ratio)
    #     x = self._skiparse_cp_scatter(x)
    #     return x

    # def _reduce(self, x, grid_sizes=None):
    #     x = self._skiparse_cp_gather(x)
    #     x = reduce(x, '(b p) n c -> b n c', 'mean', p=self.sparse_ratio)
    #     return x

    # def _skiparse_1d_single(self, x, grid_sizes=None):
    #     return rearrange(x, 'b (n p) c -> (b p) n c', p=self.sparse_ratio)

    # def _skiparse_1d_single_reverse(self, x, grid_sizes=None):
    #     return rearrange(x, '(b p) n c -> b (n p) c', p=self.sparse_ratio)

    # def _skiparse_1d_group(self, x, grid_sizes=None):
    #     return rearrange(x, 'b (n p q) c -> (b p) (n q) c', p=self.sparse_ratio, q=self.sparse_ratio)

    # def _skiparse_1d_group_reverse(self, x, grid_sizes=None):
    #     return rearrange(x, '(b p) (n q) c -> b (n p q) c', p=self.sparse_ratio, q=self.sparse_ratio)

    # single2group和group2single是完全互逆的，代码实现上是相同的
    # def _skiparse_1d_single_to_group(self, x, grid_sizes=None):
    #     k = int(self.sparse_ratio ** 0.5)
    #     return rearrange(x, '(b p q) (n r s) c -> (b r s) (n p q) c', p=k, q=k, r=k, s=k)

    # def _skiparse_1d_group_to_single(self, x, grid_sizes=None):
    #     k = int(self.sparse_ratio ** 0.5)
    #     return rearrange(x, '(b r s) (n p q) c -> (b p q) (n r s) c', p=k, q=k, r=k, s=k)

    # def _skiparse_2d_single(self, x, grid_sizes):
    #     T, H, W = grid_sizes
    #     return rearrange(x, 'b (t h p w q) c -> (b p q) (t h w) c', p=self.sparse_ratio, q=self.sparse_ratio, h=H // self.sparse_ratio, w=W // self.sparse_ratio)

    # def _skiparse_2d_single_reverse(self, x, grid_sizes):
    #     T, H, W = grid_sizes
    #     return rearrange(x, '(b p q) (t h w) c -> b (t h p w q) c', p=self.sparse_ratio, q=self.sparse_ratio, h=H // self.sparse_ratio, w=W // self.sparse_ratio)

    # def _skiparse_2d_group(self, x, grid_sizes):
    #     T, H, W = grid_sizes
    #     return rearrange(
    #         x, 'b (t h p1 p2 w q1 q2) c -> (b p1 q1) (t h p2 w q2) c',
    #         p1=self.sparse_ratio, q1=self.sparse_ratio, p2=self.sparse_ratio, q2=self.sparse_ratio, h=H // (self.sparse_ratio ** 2), w=W // (self.sparse_ratio ** 2)
    #     )

    # def _skiparse_2d_group_reverse(self, x, grid_sizes):
    #     T, H, W = grid_sizes
    #     return rearrange(
    #         x, '(b p1 q1) (t h p2 w q2) c -> b (t h p1 p2 w q1 q2) c',
    #         p1=self.sparse_ratio, q1=self.sparse_ratio, p2=self.sparse_ratio, q2=self.sparse_ratio, h=H // (self.sparse_ratio ** 2), w=W // (self.sparse_ratio ** 2)
    #     )

    # single2group和group2single是完全互逆的，代码实现上是相同的
    # def _skiparse_2d_single_to_group(self, x, grid_sizes):
    #     T, H, W = grid_sizes
    #     return rearrange(
    #         x, '(b p2 q2) (t h_p1 p1 w_q1 q1) c -> (b p1 q1) (t h_p1 p2 w_q1 q2) c',
    #         p1=self.sparse_ratio, q1=self.sparse_ratio, p2=self.sparse_ratio, q2=self.sparse_ratio, h_p1=H // (self.sparse_ratio ** 2), w_q1=W // (self.sparse_ratio ** 2)
    #     )

    # def _skiparse_2d_group_to_single(self, x, grid_sizes):
    #     T, H, W = grid_sizes
    #     return rearrange(
    #         x, '(b p1 q1) (t h_p1 p2 w_q1 q2) c -> (b p2 q2) (t h_p1 p1 w_q1 q1) c',
    #         p1=self.sparse_ratio, q1=self.sparse_ratio, p2=self.sparse_ratio, q2=self.sparse_ratio, h_p1=H // (self.sparse_ratio ** 2), w_q1=W // (self.sparse_ratio ** 2)
    #     )

    # ============ torch native实现 ============
    def _repeat(self, x, grid_sizes=None):
        """repeat: 'b n c -> (b p) n c'"""
        B, N, C = x.shape
        P = self.sparse_ratio
        # unsqueeze → expand → reshape，expand 不分配内存
        x = x.unsqueeze(1).expand(B, P, N, C).reshape(B * P, N, C)
        x = self._skiparse_cp_scatter(x)
        return x

    def _reduce(self, x, grid_sizes=None):
        """reduce mean: '(b p) n c -> b n c'"""
        x = self._skiparse_cp_gather(x)
        P = self.sparse_ratio
        BP, N, C = x.shape
        B = BP // P
        x = x.view(B, P, N, C).mean(dim=1)
        return x

    def _skiparse_1d_single(self, x, grid_sizes=None):
        """'b (n p) c -> (b p) n c'"""
        B, NP, C = x.shape
        P = self.sparse_ratio
        N = NP // P
        #   [B, N, P, C]  →  [B, P, N, C]  →  [B*P, N, C]
        return x.view(B, N, P, C).permute(0, 2, 1, 3).reshape(B * P, N, C)

    def _skiparse_1d_single_reverse(self, x, grid_sizes=None):
        """'(b p) n c -> b (n p) c'"""
        BP, N, C = x.shape
        P = self.sparse_ratio
        B = BP // P
        #   [B, P, N, C]  →  [B, N, P, C]  →  [B, N*P, C]
        return x.view(B, P, N, C).permute(0, 2, 1, 3).reshape(B, N * P, C)

    def _skiparse_1d_group(self, x, grid_sizes=None):
        """'b (n p q) c -> (b p) (n q) c'"""
        B, NPQ, C = x.shape
        P = self.sparse_ratio
        Q = self.sparse_ratio
        N = NPQ // (P * Q)
        #   [B, N, P, Q, C]  →  [B, P, N, Q, C]  →  [B*P, N*Q, C]
        return x.view(B, N, P, Q, C).permute(0, 2, 1, 3, 4).reshape(B * P, N * Q, C)

    def _skiparse_1d_group_reverse(self, x, grid_sizes=None):
        """'(b p) (n q) c -> b (n p q) c'"""
        BP, NQ, C = x.shape
        P = self.sparse_ratio
        Q = self.sparse_ratio
        B = BP // P
        N = NQ // Q
        #   [B, P, N, Q, C]  →  [B, N, P, Q, C]  →  [B, N*P*Q, C]
        return x.view(B, P, N, Q, C).permute(0, 2, 1, 3, 4).reshape(B, N * P * Q, C)

    def _skiparse_1d_single_to_group(self, x, grid_sizes=None):
        """'(b p q) (n r s) c -> (b r s) (n p q) c'"""
        k2 = self.sparse_ratio          # k² = sparse_ratio
        Bk2, Nk2, C = x.shape
        B = Bk2 // k2
        N = Nk2 // k2
        return (x.view(B, k2, N, k2, C)
                .transpose(1, 3)
                .contiguous()
                .view(Bk2, Nk2, C))

    def _skiparse_1d_group_to_single(self, x, grid_sizes=None):
        """'(b r s) (n p q) c -> (b p q) (n r s) c'"""
        return self._skiparse_1d_single_to_group(x, grid_sizes)

    def _skiparse_2d_single(self, x, grid_sizes):
        """'b (t h p w q) c -> (b p q) (t h w) c'"""
        T, H, W = grid_sizes
        B, _, C = x.shape
        P = self.sparse_ratio
        Q = self.sparse_ratio
        h = H // P
        w = W // Q
        # 展开:    [B, T, h, P, w, Q, C]   (indices: 0,1,2,3,4,5,6)
        # 目标:    [B, P, Q, T, h, w, C]
        # permute: (0, 3, 5, 1, 2, 4, 6)
        return (x.view(B, T, h, P, w, Q, C)
                .permute(0, 3, 5, 1, 2, 4, 6)
                .reshape(B * P * Q, T * h * w, C))

    def _skiparse_2d_single_reverse(self, x, grid_sizes):
        """'(b p q) (t h w) c -> b (t h p w q) c'"""
        T, H, W = grid_sizes
        P = self.sparse_ratio
        Q = self.sparse_ratio
        h = H // P
        w = W // Q
        BPQ, _, C = x.shape
        B = BPQ // (P * Q)
        # 展开:    [B, P, Q, T, h, w, C]   (indices: 0,1,2,3,4,5,6)
        # 目标:    [B, T, h, P, w, Q, C]
        # permute: (0, 3, 4, 1, 5, 2, 6)
        return (x.view(B, P, Q, T, h, w, C)
                .permute(0, 3, 4, 1, 5, 2, 6)
                .reshape(B, T * H * W, C))

    def _skiparse_2d_group(self, x, grid_sizes):
        """'b (t h p1 p2 w q1 q2) c -> (b p1 q1) (t h p2 w q2) c'"""
        T, H, W = grid_sizes
        B, _, C = x.shape
        P = self.sparse_ratio
        P2 = P * P
        h = H // P2
        w = W // P2
        # 展开:    [B, T, h, p1, p2, w, q1, q2, C]  (indices: 0,1,2,3,4,5,6,7,8)
        # 目标:    [B, p1, q1, T, h, p2, w, q2, C]
        # permute: (0, 3, 6, 1, 2, 4, 5, 7, 8)
        return (x.view(B, T, h, P, P, w, P, P, C)
                .permute(0, 3, 6, 1, 2, 4, 5, 7, 8)
                .reshape(B * P * P, T * h * P * w * P, C))

    def _skiparse_2d_group_reverse(self, x, grid_sizes):
        """'(b p1 q1) (t h p2 w q2) c -> b (t h p1 p2 w q1 q2) c'"""
        T, H, W = grid_sizes
        P = self.sparse_ratio
        P2 = P * P
        h = H // P2
        w = W // P2
        BP1Q1, _, C = x.shape
        B = BP1Q1 // (P * P)
        # 展开:    [B, p1, q1, T, h, p2, w, q2, C]  (indices: 0,1,2,3,4,5,6,7,8)
        # 目标:    [B, T, h, p1, p2, w, q1, q2, C]
        # permute: (0, 3, 4, 1, 5, 6, 2, 7, 8)
        return (x.view(B, P, P, T, h, P, w, P, C)
                .permute(0, 3, 4, 1, 5, 6, 2, 7, 8)
                .reshape(B, T * H * W, C))

    def _normal_skiparse_2d_single_to_group(self, x, grid_sizes):
        """'(b p2 q2) (t h_p1 p1 w_q1 q1) c -> (b p1 q1) (t h_p1 p2 w_q1 q2) c'"""
        T, H, W = grid_sizes
        P = self.sparse_ratio
        P2 = P * P
        h_p1 = H // P2
        w_q1 = W // P2
        BP2Q2, _, C = x.shape
        B = BP2Q2 // (P * P)
        # 展开:    [B, p2, q2, T, h_p1, p1, w_q1, q1, C]  (indices: 0,1,2,3,4,5,6,7,8)
        # 目标:    [B, p1, q1, T, h_p1, p2, w_q1, q2, C]
        # permute: (0, 5, 7, 3, 4, 1, 6, 2, 8)
        return (x.view(B, P, P, T, h_p1, P, w_q1, P, C)
                .permute(0, 5, 7, 3, 4, 1, 6, 2, 8)
                .reshape(B * P * P, T * h_p1 * P * w_q1 * P, C))

    # 采用all_to_all_single实现，通信量更小
    def _parallel_skiparse_2d_single_to_group(self, x, grid_sizes):
        T, H, W = grid_sizes
        P = self.sparse_ratio
        sub_grid_sizes = (T, H // P, W // P)
        P2 = P * P
        x = self._skiparse_2d_single(x, sub_grid_sizes)
        B, _, C = x.shape
        x = x.view(B // P2, P2, -1, C).transpose(0, 1)
        out_x = contiguous(torch.empty_like(x))
        dist.all_to_all_single(out_x, contiguous(x), group=cp_state.skiparse_cp_group)
        out_x = contiguous(out_x.transpose(0, 1)).view(B, -1, C)
        out_x = self._skiparse_2d_single_reverse(out_x, sub_grid_sizes)
        return out_x

    def _skiparse_2d_single_to_group(self, x, grid_sizes):
        if use_skiparse_context_parallel():
            return self._parallel_skiparse_2d_single_to_group(x, grid_sizes)
        # gather -> rearrange -> scatter的实现，通信量比较大，且需要大块内存
        x = self._skiparse_cp_gather(x)
        x = self._normal_skiparse_2d_single_to_group(x, grid_sizes)
        x = self._skiparse_cp_scatter(x)
        return x

    def _skiparse_2d_group_to_single(self, x, grid_sizes):
        return self._skiparse_2d_single_to_group(x, grid_sizes)
    
    def get_num_padding_tokens(self, grid_sizes):

        key = (grid_sizes, self.sparse_ratio, self.rearrange_type)
        if self.padding_cache.is_exist(key):
            return self.padding_cache.get(key)
        
        paddings = (0, 0)

        if self.skiparse_1d:
            block_size = self.sparse_ratio ** 2
            HxW = math.prod(grid_sizes[1:])
            num_padding_tokens = (block_size - HxW % block_size) % block_size
            paddings = (num_padding_tokens, 0)
        elif self.skiparse_2d:
            block_size = self.sparse_ratio ** 2
            num_padding_tokens_h = (block_size - grid_sizes[1] % block_size) % block_size
            num_padding_tokens_w = (block_size - grid_sizes[2] % block_size) % block_size
            paddings = (num_padding_tokens_h, num_padding_tokens_w)

        self.padding_cache.set(key, paddings)
        return paddings
        
    def __call__(self, x, grid_sizes=None):
        """
        带padding的skiparse rearrange。
        对于skiparse 1d，在兼容single和group的情况下padding应该为 sparse_ratio ** 2 - (H * W) % (sparse_ratio ** 2)。
        因为每sparse_ratio ** 2个token是一个子重复模式。
        例如当H * W = 10 * 10 = 100，sparse_ratio = 4时，single的重复模式是4个token，group的重复模式是4 * 4 = 16个token。
        同时兼容single和group，就需要padding = 16 - 100 % 16 = 16 - 4 = 12。
        对于skiparse 2d，可以视为是H和W两个方向上各自的skiparse 1d。
        """

        if x is None:
            return x

        if self.rearrange_type in [RearrangeType.Identity, RearrangeType.Repeat, RearrangeType.Reduce]:
            return self.rearrange_func(x, grid_sizes)
    
        B, N, C = x.shape
        x = contiguous(x)
        if RearrangeType.input_is_full(self.rearrange_type) and RearrangeType.output_is_skiparse_1d(self.rearrange_type):
            T = grid_sizes[0]
            num_padding_tokens, _ = self.get_num_padding_tokens(grid_sizes)
            if num_padding_tokens > 0:
                x = x.view(B, T, -1, C)
                padding = (0, 0, 0, num_padding_tokens)
                x = F.pad(x, padding, mode="constant", value=0).view(B, -1, C)
            x = self.rearrange_func(x, None)
            # rearrange之后，如果开启skiparse cp，则将batch维度切分到不同的skiparse cp rank
            x = self._skiparse_cp_scatter(x)
        elif RearrangeType.input_is_full(self.rearrange_type) and RearrangeType.output_is_skiparse_2d(self.rearrange_type):
            assert grid_sizes is not None and len(grid_sizes) == 3, "grid_sizes should be a tuple of (T, H, W)"
            T, H, W = grid_sizes
            num_padding_tokens_h, num_padding_tokens_w = self.get_num_padding_tokens(grid_sizes)
            if num_padding_tokens_h > 0 or num_padding_tokens_w > 0:
                x = x.view(B, T, H, W, C)
                padding = (0, 0, 0, num_padding_tokens_w, 0, num_padding_tokens_h)
                x = F.pad(x, padding, mode="constant", value=0).view(B, -1, C)
                grid_sizes = (T, H + num_padding_tokens_h, W + num_padding_tokens_w)
            x = self.rearrange_func(x, grid_sizes)
            # rearrange之后，如果开启skiparse cp，则将batch维度切分到不同的skiparse cp rank
            x = self._skiparse_cp_scatter(x)
        elif RearrangeType.input_is_skiparse_1d(self.rearrange_type) and RearrangeType.output_is_full(self.rearrange_type):
            # 如果skiparse cp开启且输入为skiparse后的序列，则先在batch维度gather
            x = self._skiparse_cp_gather(x)
            x = self.rearrange_func(x, None)
            B = x.shape[0]
            T = grid_sizes[0]
            num_padding_tokens, _ = self.get_num_padding_tokens(grid_sizes)
            if num_padding_tokens > 0:
                x = x.view(B, T, -1, C)
                x = x[:, :, :-num_padding_tokens]
                x = contiguous(x).view(B, -1, C)
        elif RearrangeType.input_is_skiparse_2d(self.rearrange_type) and RearrangeType.output_is_full(self.rearrange_type):
            # 如果skiparse cp开启且输入为skiparse后的序列，则先在batch维度gather
            x = self._skiparse_cp_gather(x)
            T, H, W = grid_sizes
            num_padding_tokens_h, num_padding_tokens_w = self.get_num_padding_tokens(grid_sizes)
            if num_padding_tokens_h > 0 or num_padding_tokens_w > 0:
                H = H + num_padding_tokens_h
                W = W + num_padding_tokens_w
                grid_sizes = (T, H, W)
            x = self.rearrange_func(x, grid_sizes)
            B = x.shape[0]
            if num_padding_tokens_h > 0 or num_padding_tokens_w > 0:
                H_orig = H - num_padding_tokens_h  # 注意这里 H 已经是 padded 的
                W_orig = W - num_padding_tokens_w
                x = contiguous(x.view(B, T, H, W, C)[:, :, :H_orig, :W_orig]).view(B, -1, C)
        # 如果是single2group或者group2single，则不需要重新padding
        elif RearrangeType.is_single2group(self.rearrange_type) or RearrangeType.is_group2single(self.rearrange_type):
            if self.skiparse_2d:
                T, H, W = grid_sizes
                num_padding_tokens_h, num_padding_tokens_w = self.get_num_padding_tokens(grid_sizes)
                grid_sizes = (T, H + num_padding_tokens_h, W + num_padding_tokens_w)
            x = self.rearrange_func(x, grid_sizes)
        return x


class MetaPreprocessor(ABC):
    def __init__(
        self,
        is_skiparse_1d_model=False,
        is_skiparse_2d_model=False,
        sparse_ratio=4,
    ):
        self.is_skiparse_1d_model = is_skiparse_1d_model
        self.is_skiparse_2d_model = is_skiparse_2d_model
        self.sparse_ratio = sparse_ratio
        if self.is_skiparse_1d_model and self.is_skiparse_2d_model:
            raise ValueError(f"We can only use skiparse 1d or skiparse 2d, not both at the same time!")
        if (not self.is_skiparse_1d_model and not self.is_skiparse_2d_model) and self.sparse_ratio > 1:
            warnings.warn("When skiparse_1d = skiparse_2d = False, sparse ratio should be 1, we instead use full attention.")
            self.sparse_ratio = 1
    
    @abstractmethod
    def preprocess(self, x, grid_sizes, **kwargs):
        pass
            
class ContextParallelPreprocessor(MetaPreprocessor):
    """
    在开启context parallel时，正常的skiparse会失效，因为skiparse需要获取全序列信息。
    一个合理的思路是，按照某种既定规则划分sequence，使得在序列内做skiparse等价于在全序列上做skiparse
    """

    def __init__(
        self,
        is_skiparse_1d_model=False,
        is_skiparse_2d_model=False,
        sparse_ratio=4,
    ):
        super().__init__(is_skiparse_1d_model, is_skiparse_2d_model, sparse_ratio)
        self.shard_seq_lens_cache = SafeCacheManager()

    def check_short_sequence(self, num_tokens_or_sub_sequences: int, cp_size: int):
        if (num_tokens_or_sub_sequences % cp_size != 0 and
                num_tokens_or_sub_sequences <= (num_tokens_or_sub_sequences // cp_size + 1) * (cp_size - 1)):
            raise ValueError(
                f"Token {num_tokens_or_sub_sequences} is too short to be divided into {cp_size} parts"
            )

    def _skiparse_1d_params(self, H, W, cp_size):
        sub_len = self.sparse_ratio ** 2                      # 每个子模式长度
        num_sub = math.ceil(H * W / sub_len)                  # 子模式个数
        self.check_short_sequence(num_sub, cp_size)
        seq_len_per_cp = math.ceil(num_sub / cp_size) * sub_len
        return sub_len, num_sub, seq_len_per_cp

    def _skiparse_2d_params(self, H, W, cp_size):
        # 尽可能接近正方形，如果无法达到正方形，则让cp_size_w更大（因为一般用横屏视频）
        cp_size_h = cp_size // math.ceil(cp_size ** 0.5)
        cp_size_w = cp_size // cp_size_h
        sub_h = self.sparse_ratio ** 2
        sub_w = self.sparse_ratio ** 2
        num_sub_h = math.ceil(H / sub_h)
        num_sub_w = math.ceil(W / sub_w)
        self.check_short_sequence(num_sub_h, cp_size_h)
        self.check_short_sequence(num_sub_w, cp_size_w)
        seq_h = math.ceil(num_sub_h / cp_size_h) * sub_h
        seq_w = math.ceil(num_sub_w / cp_size_w) * sub_w
        return sub_h, sub_w, num_sub_h, num_sub_w, seq_h, seq_w

    # ----------------- preprocess -----------------

    def preprocess(self, x, grid_sizes):
        if not use_context_parallel():
            return x, grid_sizes

        x = contiguous(x)
        B, N, C = x.shape
        T, H, W = grid_sizes
        cp_size = cp_state.cp_size
        cp_rank = cp_state.cp_rank
        sub_grid_sizes = grid_sizes

        # skiparse 1d
        if self.is_skiparse_1d_model:
            x = x.view(B, T, H * W, C)
            _, _, seq_hw = self._skiparse_1d_params(H, W, cp_size)

            start = cp_rank * seq_hw
            assert start < H * W, "The start index should be less than the total number of tokens"
            end = min(start + seq_hw, H * W)
            x = x[:, :, start:end, :]              # [B, T, seq_hw, C]
            sub_grid_sizes = (T, end - start)
            return contiguous(x).view(B, -1, C), sub_grid_sizes

        # skiparse 2d
        if self.is_skiparse_2d_model:
            x = x.view(B, T, H, W, C)
            _, _, _, _, seq_h, seq_w = self._skiparse_2d_params(H, W, cp_size)

            # 尽可能接近正方形，如果无法达到正方形，则让cp_size_w更大（因为一般用横屏视频）
            cp_size_h = cp_size // math.ceil(cp_size ** 0.5)
            cp_size_w = cp_size // cp_size_h
            index_h = cp_rank // cp_size_w
            index_w = cp_rank % cp_size_w
            start_h = index_h * seq_h
            assert start_h < H, "The start index should be less than the height"
            end_h = min(start_h + seq_h, H)
            start_w = index_w * seq_w
            assert start_w < W, "The start index should be less than the width"
            end_w = min(start_w + seq_w, W)
            x = x[:, :, start_h:end_h, start_w:end_w, :]  # [B, T, seq_h, seq_w, C]
            sub_grid_sizes = (T, end_h - start_h, end_w - start_w)
            return contiguous(x).view(B, -1, C), sub_grid_sizes

        # 普通 cp
        self.check_short_sequence(N, cp_size)
        return contiguous(torch.chunk(x, cp_size, dim=1)[cp_state.cp_rank]), sub_grid_sizes

    # ----------------- postprocess -----------------

    def postprocess(self, x, grid_sizes, shard_seq_lens=None):
        if not use_context_parallel():
            return x

        x = contiguous(x)
        T, H, W = grid_sizes
        cp_size = cp_state.cp_size

        # ========== skiparse 1d 逆 ==========
        if self.is_skiparse_1d_model:
            _, _, seq_hw = self._skiparse_1d_params(H, W, cp_size)

            B, _, C = x.shape
            x = all_gather(x, dim=1, group=cp_state.cp_group)
            x_list = x.split_with_sizes(shard_seq_lens, dim=1)

            full = torch.empty(B, T, H * W, C, device=x.device, dtype=x.dtype)

            for r in range(cp_size):
                start = r * seq_hw
                end = min(start + seq_hw, H * W)
                full[:, :, start:end, :] = x_list[r].view(B, T, end - start, C)

            return contiguous(full).view(B, -1, C)

        # ========== skiparse 2d 逆 ==========
        if self.is_skiparse_2d_model:
            _, _, _, _, seq_h, seq_w = self._skiparse_2d_params(H, W, cp_size)
            cp_size_h = cp_size // math.ceil(cp_size ** 0.5)
            cp_size_w = cp_size // cp_size_h

            B, _, C = x.shape
            x = all_gather(x, dim=1, group=cp_state.cp_group)
            x_list = x.split_with_sizes(shard_seq_lens, dim=1)
            full = torch.empty(B, T, H, W, C, device=x.device, dtype=x.dtype)

            for r in range(cp_size):
                index_h = r // cp_size_w
                index_w = r % cp_size_w
                start_h = index_h * seq_h
                end_h = min(start_h + seq_h, H)
                start_w = index_w * seq_w
                end_w = min(start_w + seq_w, W)

                h_len = end_h - start_h
                w_len = end_w - start_w
                full[:, :, start_h:end_h, start_w:end_w, :] = x_list[r].view(B, T, h_len, w_len, C)

            return contiguous(full).view(B, -1, C)

        # ========== 普通 cp 逆 ==========
        return all_gather(x, dim=1, group=cp_state.cp_group)

    def get_shard_seq_lens(self, shape, grid_sizes, device="cuda"):
        if not use_context_parallel():
            return [shape[1]] * 3

        key = (shape, grid_sizes)
        if self.shard_seq_lens_cache.is_exist(key):
            return self.shard_seq_lens_cache.get(key)
        
        _, N, _ = shape
        
        dummy = torch.ones((1, N, 1), dtype=torch.bool, device=device)
        dummy, sub_grid_sizes = self.preprocess(dummy, grid_sizes)

        full_shard_seq_lens = get_shard_seq_lens(dummy, cp_state.cp_group)

        if self.is_skiparse_1d_model:
            single_rearrange_type = RearrangeType.Skiparse1DSingle
            group_rearrange_type = RearrangeType.Skiparse1DGroup
        elif self.is_skiparse_2d_model:
            single_rearrange_type = RearrangeType.Skiparse2DSingle
            group_rearrange_type = RearrangeType.Skiparse2DGroup
        else:
            single_rearrange_type = RearrangeType.Identity
            group_rearrange_type = RearrangeType.Identity
        single_rearrange = SkiparseRearrange(self.sparse_ratio, single_rearrange_type)
        group_rearrange = SkiparseRearrange(self.sparse_ratio, group_rearrange_type)
        single_dummy = single_rearrange(dummy, sub_grid_sizes)
        group_dummy = group_rearrange(dummy, sub_grid_sizes)
        single_shard_seq_lens = get_shard_seq_lens(single_dummy, cp_state.cp_group)
        group_shard_seq_lens = get_shard_seq_lens(group_dummy, cp_state.cp_group)

        self.shard_seq_lens_cache.set(key, (full_shard_seq_lens, single_shard_seq_lens, group_shard_seq_lens))

        return full_shard_seq_lens, single_shard_seq_lens, group_shard_seq_lens


class SkiparseMaskPreprocessor(MetaPreprocessor):
    """
        生成skiparse attention下的attention mask，考虑采用register tokens的情况。
    """

    def __init__(
        self, 
        is_skiparse_1d_model=False,
        is_skiparse_2d_model=False,
        sparse_ratio=4,
        num_register_tokens=1024,
    ):
        super().__init__(is_skiparse_1d_model, is_skiparse_2d_model, sparse_ratio)
        self.num_register_tokens = num_register_tokens
        self.cache = SafeCacheManager()

        if self.is_skiparse_1d_model:
            self.single_rearrange_type = RearrangeType.Skiparse1DSingle
            self.group_rearrange_type = RearrangeType.Skiparse1DGroup
        elif self.is_skiparse_2d_model:
            self.single_rearrange_type = RearrangeType.Skiparse2DSingle
            self.group_rearrange_type = RearrangeType.Skiparse2DGroup
        else:
            self.single_rearrange_type = RearrangeType.Identity
            self.group_rearrange_type = RearrangeType.Identity
        self.single_rearrange = SkiparseRearrange(self.sparse_ratio, self.single_rearrange_type)
        self.group_rearrange = SkiparseRearrange(self.sparse_ratio, self.group_rearrange_type)

    def preprocess(self, shape, grid_sizes, context_preprocessor=None, dtype=torch.bool, device="cuda"):
        if (not self.is_skiparse_1d_model and not self.is_skiparse_2d_model) or self.sparse_ratio == 1:
            return None, None

        key = (shape, grid_sizes)
        if self.cache.is_exist(key):
            return self.cache.get(key)

        B, N, _ = shape
        mask = torch.ones((B, N, 1), dtype=dtype, device=device)
        sub_grid_sizes = grid_sizes

        if use_context_parallel() and context_preprocessor is not None:
            mask, sub_grid_sizes = context_preprocessor.preprocess(mask, grid_sizes)

        orig_single_mask = self.single_rearrange(mask, sub_grid_sizes)
        orig_group_mask = self.group_rearrange(mask, sub_grid_sizes)

        if use_context_parallel() and context_preprocessor is not None:
            stacked = torch.cat([orig_single_mask, orig_group_mask], dim=-1)  # [B, N, 2]
            stacked = all_gather(stacked, dim=1, group=cp_state.cp_group)
            orig_single_mask, orig_group_mask = stacked.split(1, dim=-1)

        single_mask = orig_single_mask.squeeze(-1).bool()
        group_mask = orig_group_mask.squeeze(-1).bool()

        if self.num_register_tokens > 0:
            single_mask = F.pad(single_mask, (self.num_register_tokens, 0), value=True)
            group_mask = F.pad(group_mask, (self.num_register_tokens, 0), value=True)

        single_mask = contiguous(single_mask)
        group_mask = contiguous(group_mask)

        self.cache.set(key, (single_mask, group_mask))
        return single_mask, group_mask

class SkiparseRopeWrapper:
    def __init__(self, freqs, context_preprocessor=None):
        self.freqs = freqs
        self.context_preprocessor = context_preprocessor
        # 最大缓存4个不同的rearrange type的rope cache
        # 实际上一个model最多有3个rope rearrange type: indentity, single, group
        self.cache = SafeCacheManager(max_cache_size=4)
        # 用float32和complex64精度足够
        self.real_dtype = torch.float32
        self.complex_dtype = torch.complex64

    # ----------------- skiparse rope -----------------
    @torch.autocast("cuda", enabled=False)
    def apply_rope(self, x, grid_sizes, skiparse_rerrange=None):
        seq_len, num_heads, head_dim = x.size(1), x.size(2), x.size(3) // 2
        x = torch.view_as_complex(
            x.to(self.real_dtype).reshape(-1, seq_len, num_heads, head_dim, 2)
        )
        T, H, W = grid_sizes
        rearrange_type = skiparse_rerrange.rearrange_type if skiparse_rerrange is not None else RearrangeType.Identity
        key = (T, H, W, rearrange_type)
        if self.cache.is_exist(key):
            freqs_i = self.cache.get(key)
        else:
            # split freqs
            freqs = self.freqs.split([head_dim - 2 * (head_dim // 3), head_dim // 3, head_dim // 3], dim=1)
            freqs = tuple(f.to(self.complex_dtype) for f in freqs)

            freqs_i = torch.cat(
                [
                    freqs[0][:T].view(T, 1, 1, -1).expand(T, H, W, -1),
                    freqs[1][:H].view(1, H, 1, -1).expand(T, H, W, -1),
                    freqs[2][:W].view(1, 1, W, -1).expand(T, H, W, -1),
                ],
                dim=-1,
            ).reshape(1, T * H * W, -1)

            sub_grid_sizes = grid_sizes
            if use_context_parallel() and self.context_preprocessor is not None:
                # 按照context parallel规则切分得到子序列的freqs
                freqs_i, sub_grid_sizes = self.context_preprocessor.preprocess(freqs_i, grid_sizes)

            if skiparse_rerrange is not None:
                # 如果采用skiparse，则对子序列的freqs进行rearrange
                freqs_i = skiparse_rerrange(freqs_i, sub_grid_sizes)

            if use_context_parallel() and self.context_preprocessor is not None:
                # 因为rope是对all2all后的序列添加，所以需要将子序列的freqs再all gather到全局的freqs
                freqs_i = all_gather(freqs_i, dim=1, group=cp_state.cp_group)

            freqs_i = freqs_i.unsqueeze(2)
            self.cache.set(key, freqs_i)
        # apply rotary embedding
        x = torch.view_as_real(x * freqs_i).flatten(3)
        return x.float()


class OSPNextSelfAttention(nn.Module):

    def __init__(
        self, 
        dim, 
        num_heads, 
        window_size=(-1, -1), 
        qk_norm=True, 
        eps=1e-6, 
        # skiparse相关
        sparse_ratio=4,
        skiparse_1d=False,
        skiparse_2d=False,
        skiparse_block_type=SkiparseBlockType.Full,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.skiparse_1d = skiparse_1d
        self.skiparse_2d = skiparse_2d

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = OSPNextRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = OSPNextRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        if skiparse_block_type == SkiparseBlockType.Full:
            self.rearrange_rope = SkiparseRearrange(rearrange_type=RearrangeType.Identity)
        elif skiparse_block_type == SkiparseBlockType.Single:
            self.rearrange_rope = SkiparseRearrange(
                sparse_ratio=sparse_ratio,
                rearrange_type=RearrangeType.Skiparse1DSingle if self.skiparse_1d else RearrangeType.Skiparse2DSingle,
            )
        elif skiparse_block_type == SkiparseBlockType.Group:
            self.rearrange_rope = SkiparseRearrange(
                sparse_ratio=sparse_ratio,
                rearrange_type=RearrangeType.Skiparse1DGroup if self.skiparse_1d else RearrangeType.Skiparse2DGroup,
            )

    def forward(
        self, 
        x, 
        attn_mask,
        grid_sizes_for_rope,
        rope_wrapper,
        num_register_tokens=0, 
        shard_seq_lens=None,
    ):
        B, N, H, D = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(B, N, H, D)
        k = self.norm_k(self.k(x)).view(B, N, H, D)
        v = self.v(x).view(B, N, H, D)

        if num_register_tokens > 0:
            register_q, q = q.split_with_sizes([num_register_tokens, N - num_register_tokens], dim=1)
            register_k, k = k.split_with_sizes([num_register_tokens, N - num_register_tokens], dim=1)
            register_v, v = v.split_with_sizes([num_register_tokens, N - num_register_tokens], dim=1)

        if use_context_parallel():
            if num_register_tokens > 0:
                register_q = torch.chunk(register_q, cp_state.cp_size, dim=2)[cp_state.cp_rank]
                register_k = torch.chunk(register_k, cp_state.cp_size, dim=2)[cp_state.cp_rank]
                register_v = torch.chunk(register_v, cp_state.cp_size, dim=2)[cp_state.cp_rank]
            q = all_to_all_4D(q, group=cp_state.cp_group, scatter_dim=2, gather_dim=1, shard_seq_lens=shard_seq_lens)
            k = all_to_all_4D(k, group=cp_state.cp_group, scatter_dim=2, gather_dim=1, shard_seq_lens=shard_seq_lens)
            v = all_to_all_4D(v, group=cp_state.cp_group, scatter_dim=2, gather_dim=1, shard_seq_lens=shard_seq_lens)

        q = rope_wrapper.apply_rope(q, grid_sizes_for_rope, self.rearrange_rope)
        k = rope_wrapper.apply_rope(k, grid_sizes_for_rope, self.rearrange_rope)

        if num_register_tokens > 0:
            q = torch.cat([register_q, q], dim=1)   
            k = torch.cat([register_k, k], dim=1)
            v = torch.cat([register_v, v], dim=1)
        
        x = attention_with_mask(
            q,
            k, 
            v, 
            attn_mask
        )

        if use_context_parallel():
            if num_register_tokens > 0:
                register_x, x = x.split_with_sizes([num_register_tokens, N - num_register_tokens], dim=1)
                register_x = all_gather(register_x, dim=2, group=cp_state.cp_group)
            x = all_to_all_4D(x, group=cp_state.cp_group, scatter_dim=1, gather_dim=2, shard_seq_lens=shard_seq_lens)
            if num_register_tokens > 0:
                x = torch.cat([register_x, x], dim=1)

        # output
        x = x.flatten(2)
        x = self.o(x)

        return x


class OSPNextCrossAttention(OSPNextSelfAttention):

    def forward(
        self, 
        x, 
        text,
        num_register_tokens=0,
        shard_seq_lens=None,
    ):

        B, N, H, D = *x.shape[:2], self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(B, -1, H, D)
        k = self.norm_k(self.k(text)).view(B, -1, H, D)
        v = self.v(text).view(B, -1, H, D)

        if use_context_parallel():
            if num_register_tokens > 0:
                register_q, q = q.split_with_sizes([num_register_tokens, N - num_register_tokens], dim=1)
            q = all_to_all_4D(q, group=cp_state.cp_group, scatter_dim=2, gather_dim=1, shard_seq_lens=shard_seq_lens)
            if num_register_tokens > 0:
                register_q = torch.chunk(register_q, cp_state.cp_size, dim=2)[cp_state.cp_rank]
                q = torch.cat([register_q, q], dim=1)
            k = torch.chunk(k, cp_state.cp_size, dim=2)[cp_state.cp_rank]
            v = torch.chunk(v, cp_state.cp_size, dim=2)[cp_state.cp_rank]

        x = attention_with_mask(q, k, v, attn_mask=None)

        if use_context_parallel():
            if num_register_tokens > 0:
                register_x, x = x.split_with_sizes([num_register_tokens, N - num_register_tokens], dim=1)
                register_x = all_gather(register_x, dim=2, group=cp_state.cp_group)
            x = all_to_all_4D(x, group=cp_state.cp_group, scatter_dim=1, gather_dim=2, shard_seq_lens=shard_seq_lens)
            if num_register_tokens > 0:
                x = torch.cat([register_x, x], dim=1)
        
        # output
        x = x.flatten(2)
        x = self.o(x)

        return x


class OSPNextAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        # skiparse相关
        sparse_ratio=4,
        skiparse_1d=False,
        skiparse_2d=False,
        skiparse_block_type=SkiparseBlockType.Full,
        is_full2skiparse_block=False,
        is_skiparse2full_block=False,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = OSPNextLayerNorm(dim, eps)
        self.self_attn = OSPNextSelfAttention(
            dim, num_heads, window_size, qk_norm, eps,
            sparse_ratio=sparse_ratio,
            skiparse_1d=skiparse_1d,
            skiparse_2d=skiparse_2d,
            skiparse_block_type=skiparse_block_type,
        )
        self.norm3 = (
            OSPNextLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = OSPNextCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = OSPNextLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        self.skiparse_1d = skiparse_1d
        self.skiparse_2d = skiparse_2d
        self.sparse_ratio = sparse_ratio
        self.skiparse_block_type = skiparse_block_type
        self.is_skiparse2full_block = is_skiparse2full_block
        self.is_full2skiparse_block = is_full2skiparse_block

        if self.skiparse_block_type == SkiparseBlockType.Full:
            self.rearrange_input = SkiparseRearrange(rearrange_type=RearrangeType.Identity)
            self.rearrange_output = SkiparseRearrange(rearrange_type=RearrangeType.Identity)
            self.text_rearrange_input = SkiparseRearrange(rearrange_type=RearrangeType.Identity)
            self.register_rearrange_input = SkiparseRearrange(rearrange_type=RearrangeType.Identity)
            self.register_rearrange_output = SkiparseRearrange(rearrange_type=RearrangeType.Identity)
        else:
            self.text_rearrange_input = SkiparseRearrange(
                sparse_ratio=self.sparse_ratio if self.skiparse_1d else self.sparse_ratio ** 2, 
                rearrange_type=RearrangeType.Repeat
            )
            if self.skiparse_block_type == SkiparseBlockType.Single:
                if self.is_full2skiparse_block:
                    self.rearrange_input = SkiparseRearrange(
                        sparse_ratio=self.sparse_ratio,
                        rearrange_type=RearrangeType.Skiparse1DSingle if self.skiparse_1d else RearrangeType.Skiparse2DSingle,
                    )
                    self.rearrange_output = SkiparseRearrange(rearrange_type=RearrangeType.Identity)
                    self.register_rearrange_input = SkiparseRearrange(
                        sparse_ratio=self.sparse_ratio if self.skiparse_1d else self.sparse_ratio ** 2, 
                        rearrange_type=RearrangeType.Repeat
                    )
                    self.register_rearrange_output = SkiparseRearrange(rearrange_type=RearrangeType.Identity)
                else:
                    self.rearrange_input = SkiparseRearrange(
                        sparse_ratio=self.sparse_ratio,
                        rearrange_type=RearrangeType.Skiparse1DGroup2Single if self.skiparse_1d else RearrangeType.Skiparse2DGroup2Single,
                    )
                    self.rearrange_output = SkiparseRearrange(rearrange_type=RearrangeType.Identity)
                    self.register_rearrange_input = SkiparseRearrange(rearrange_type=RearrangeType.Identity)
                    self.register_rearrange_output = SkiparseRearrange(rearrange_type=RearrangeType.Identity)
            elif self.skiparse_block_type == SkiparseBlockType.Group:
                if self.is_skiparse2full_block:
                    self.rearrange_input = SkiparseRearrange(
                        sparse_ratio=self.sparse_ratio,
                        rearrange_type=RearrangeType.Skiparse1DSingle2Group if self.skiparse_1d else RearrangeType.Skiparse2DSingle2Group,
                    )
                    self.rearrange_output = SkiparseRearrange(
                        sparse_ratio=self.sparse_ratio,
                        rearrange_type=RearrangeType.Skiparse1DGroupReverse if self.skiparse_1d else RearrangeType.Skiparse2DGroupReverse,
                    )
                    self.register_rearrange_input = SkiparseRearrange(rearrange_type=RearrangeType.Identity)
                    self.register_rearrange_output = SkiparseRearrange(
                        sparse_ratio=self.sparse_ratio if self.skiparse_1d else self.sparse_ratio ** 2, 
                        rearrange_type=RearrangeType.Reduce
                    )
                else:
                    self.rearrange_input = SkiparseRearrange(
                        sparse_ratio=self.sparse_ratio,
                        rearrange_type=RearrangeType.Skiparse1DSingle2Group if self.skiparse_1d else RearrangeType.Skiparse2DSingle2Group,
                    )
                    self.rearrange_output = SkiparseRearrange(rearrange_type=RearrangeType.Identity)
                    self.register_rearrange_input = SkiparseRearrange(rearrange_type=RearrangeType.Identity)
                    self.register_rearrange_output = SkiparseRearrange(rearrange_type=RearrangeType.Identity)

    def block_forward(
        self,
        x,
        attn_mask,
        e,
        grid_sizes_for_rope,
        rope_wrapper,
        text,
        num_register_tokens=None,
        shard_seq_lens=None,
    ):
        e = (self.modulation + e).chunk(6, dim=1)

        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0], 
            attn_mask, 
            grid_sizes_for_rope, 
            rope_wrapper, 
            num_register_tokens=num_register_tokens,
            shard_seq_lens=shard_seq_lens,
        )
        x = x + y * e[2]
        # cross-attention & ffn function
        x = x + self.cross_attn(
            self.norm3(x), 
            text, 
            num_register_tokens=num_register_tokens, 
            shard_seq_lens=shard_seq_lens,
        )
        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
        x = x + y * e[5]

        return x

    def forward(
        self,
        x,
        attn_mask,
        e,
        sub_grid_sizes,
        grid_sizes_for_rope,
        rope_wrapper,
        text,
        register_tokens=None,
        shard_seq_lens=None,
        gradient_checkpointing=False,
    ):
        # 对输入做rerrange
        x = self.rearrange_input(x, grid_sizes=sub_grid_sizes)
        text = self.text_rearrange_input(text)
        register_tokens = self.register_rearrange_input(register_tokens)

        num_register_tokens = 0
        if register_tokens is not None:
            num_register_tokens = register_tokens.size(1)
            x = torch.cat([register_tokens, x], dim=1)

        if gradient_checkpointing and torch.is_grad_enabled():
            x = torch.utils.checkpoint.checkpoint(
                self.block_forward, 
                x, 
                attn_mask, 
                e, 
                grid_sizes_for_rope, 
                rope_wrapper, 
                text, 
                num_register_tokens, 
                shard_seq_lens,
                use_reentrant=False,
            )
        else:
            x = self.block_forward(
                x, 
                attn_mask, 
                e, 
                grid_sizes_for_rope, 
                rope_wrapper, 
                text, 
                num_register_tokens, 
                shard_seq_lens
            )

        if num_register_tokens > 0:
            register_tokens, x = x.split_with_sizes([num_register_tokens, x.shape[1] - num_register_tokens], dim=1)

        # 对输出做rerrange
        x = self.rearrange_output(x, grid_sizes=sub_grid_sizes)
        register_tokens = self.register_rearrange_output(register_tokens)

        return x, register_tokens


class OSPNextModel(ModelMixin, ConfigMixin):

    r"""
    OSPNext model. 基于WanT2V，添加了skiparse机制。
    """

    ignore_for_config = [
        "patch_size",
        "cross_attn_norm",
        "qk_norm",
        "text_dim",
        "window_size",
    ]

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        # skiparse相关参数
        skiparse_model_type=SkiparseModelType.Full,
        sparse_ratio=1,
        num_full_blocks=0,
        num_register_tokens=0,
        skiparse_1d=False,
        skiparse_2d=False,
        **kwargs,
    ):

        super().__init__()

        assert model_type in ["t2v", "i2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.gradient_checkpointing = False

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.skiparse_model_type = skiparse_model_type
        self.skiparse_1d = skiparse_1d
        self.skiparse_2d = skiparse_2d
        self.sparse_ratio = sparse_ratio
        self.num_full_blocks = num_full_blocks
        self.num_register_tokens = num_register_tokens

        if self.skiparse_model_type == SkiparseModelType.Full:
            self.skiparse_1d = self.skiparse_2d = False
            self.sparse_ratio = 1
            self.num_full_blocks = self.num_layers
            self.full_block_indices = list(range(0, self.num_layers))
        else:
            assert self.num_layers % 2 == 0 and self.num_full_blocks % 2 == 0 and self.num_full_blocks <= self.num_layers // 2, "num_full_blocks should be divisible by 2 and less than or equal to num_layers // 2"
            if self.skiparse_model_type == SkiparseModelType.DualEnd:
                assert self.num_full_blocks % 4 == 0, "num_full_blocks should be divisible by 4"
                skiparse_start_index = self.num_full_blocks // 2
                skiparse_end_index = self.num_layers - self.num_full_blocks // 2 - 1
                assert skiparse_start_index < skiparse_end_index, "skiparse_start_index should be less than skiparse_end_index"
                self.full_block_indices = list(range(0, skiparse_start_index)) + list(range(skiparse_end_index + 1, self.num_layers))
            elif self.skiparse_model_type == SkiparseModelType.Uniform:
                full_block_interval = self.num_layers // (self.num_full_blocks // 2)
                assert full_block_interval % 2 == 0, "full_block_interval should be divisible by 2"
                full_block_indices = list(range(0, self.num_layers, full_block_interval))
                self.full_block_indices = full_block_indices + [idx + 1 for idx in full_block_indices]
                self.full_block_indices = sorted(self.full_block_indices)[:self.num_full_blocks]
            else:
                raise ValueError(f"Unsupported skiparse model type: {self.skiparse_model_type}")

        # 添加虚拟的两端full block，方便处理边界情况
        self.full_block_indices = [-1] + self.full_block_indices + [self.num_layers]

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            if i in self.full_block_indices:
                skiparse_block_type = SkiparseBlockType.Full
            elif i % 2 == 0:
                skiparse_block_type = SkiparseBlockType.Single
            else:
                skiparse_block_type = SkiparseBlockType.Group
            is_full2skiparse_block = (i - 1) in self.full_block_indices and i not in self.full_block_indices
            is_skiparse2full_block = (i + 1) in self.full_block_indices and i not in self.full_block_indices
            self.blocks.append(
                OSPNextAttentionBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    sparse_ratio=sparse_ratio if skiparse_block_type is not SkiparseBlockType.Full else 1,
                    skiparse_1d=skiparse_1d if skiparse_block_type is not SkiparseBlockType.Full else False,
                    skiparse_2d=skiparse_2d if skiparse_block_type is not SkiparseBlockType.Full else False,
                    skiparse_block_type=skiparse_block_type,
                    is_full2skiparse_block=is_full2skiparse_block,
                    is_skiparse2full_block=is_skiparse2full_block,
                )
            )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        self.rope_d = dim // num_heads
        self.freqs = None
        self.rope_wrapper = None

        # register tokens
        # if num_register_tokens > 0, we use register tokens
        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.randn(1, self.num_register_tokens, dim))
        else:
            self.register_tokens = None

        self.mask_preprocessor = SkiparseMaskPreprocessor(
            is_skiparse_1d_model=self.skiparse_1d,
            is_skiparse_2d_model=self.skiparse_2d,
            sparse_ratio=self.sparse_ratio,
            num_register_tokens=self.num_register_tokens,
        )

        self.context_preprocessor = ContextParallelPreprocessor(
            is_skiparse_1d_model=self.skiparse_1d,
            is_skiparse_2d_model=self.skiparse_2d,
            sparse_ratio=self.sparse_ratio,
        )

        if safe_get_rank() == 0:
            print(f"=" * 20 + f"OSPNextModel init" + "=" * 20)
            print(f"skiparse_model_type: {self.skiparse_model_type}")
            print(f"skiparse_1d: {self.skiparse_1d}")
            print(f"skiparse_2d: {self.skiparse_2d}")
            print(f"sparse_ratio: {self.sparse_ratio}")
            print(f"num_full_blocks: {self.num_full_blocks}")
            print(f"num_register_tokens: {self.num_register_tokens}")
            print(f"full_block_indices: {self.full_block_indices}")
            print(f"=" * 20 + f"OSPNextModel init" + "=" * 20)

        # initialize weights
        self.init_weights()

    def set_gradient_checkpointing(self, enabled = False):
        self.gradient_checkpointing = enabled 

    def reset_parameters(self):
        print(f"{__class__.__name__} reset parameters!")
        self.init_weights()

    # lock main parameters except register tokens
    def lock_main_parameters(self):
        for name, param in self.named_parameters():
            if "register_tokens" in name:
                continue
            param.requires_grad = False

    def forward(
        self,
        x, # [B C T H W]
        t, # [B]
        text, # [B N C]
        **kwargs,
    ):

        # params
        device = self.patch_embedding.weight.device

        # maybe we use meta device for init, so rope freqs should init before forward
        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        if self.freqs is None:
            self.freqs = torch.cat(
                [
                    rope_params(1024, self.rope_d - 4 * (self.rope_d // 6)),
                    rope_params(1024, 2 * (self.rope_d // 6)),
                    rope_params(1024, 2 * (self.rope_d // 6)),
                ],
                dim=1,
            ).to(device)
            self.rope_wrapper = SkiparseRopeWrapper(self.freqs, self.context_preprocessor)

        # embeddings
        x = self.patch_embedding(x)

        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        x, grid_sizes = self.patchify(x)
        grid_sizes_for_rope = grid_sizes
        patchify_x_shape = x.shape

        # 计算shard_seq_lens，在all2all中用于恢复原序列长度
        full_shard_seq_lens, single_shard_seq_lens, group_shard_seq_lens = self.context_preprocessor.get_shard_seq_lens(
            patchify_x_shape, grid_sizes
        )

        # skiparse过程中可能有padding，所以需要生成mask
        # mask需要传入context preprocessor，以对齐cp切分后的seq
        single_mask, group_mask = self.mask_preprocessor.preprocess(
            patchify_x_shape, grid_sizes, context_preprocessor=self.context_preprocessor,
            dtype=torch.bool, device=device
        )
        
        # cp管理器。要兼容skiparse attn的情况下，Ulysses cp需要按照特定规则切分序列
        x, sub_grid_sizes = self.context_preprocessor.preprocess(x, grid_sizes)

        # text
        text = self.text_embedding(text)

        if self.num_register_tokens > 0:
            register_tokens = self.register_tokens.repeat(x.size(0), 1, 1)
        else:
            register_tokens = None

        for block in self.blocks:
            if block.skiparse_block_type == SkiparseBlockType.Full:
                attn_mask, shard_seq_lens = None, full_shard_seq_lens
            elif block.skiparse_block_type == SkiparseBlockType.Single:
                attn_mask, shard_seq_lens = single_mask, single_shard_seq_lens
            elif block.skiparse_block_type == SkiparseBlockType.Group:
                attn_mask, shard_seq_lens = group_mask, group_shard_seq_lens
            x, register_tokens = block(
                x, 
                attn_mask, 
                e0, 
                sub_grid_sizes, 
                grid_sizes_for_rope, 
                self.rope_wrapper, 
                text, 
                register_tokens, 
                shard_seq_lens, 
                self.gradient_checkpointing
            )
        # head
        x = self.head(x, e)

        x = self.context_preprocessor.postprocess(x, grid_sizes, shard_seq_lens=full_shard_seq_lens)

        # unpatchify
        x = self.unpatchify(x, *grid_sizes)
        return x.float()

    def patchify(self, embs):
        # get f, h, w from b c f h w
        grid_sizes = embs.shape[2:]

        # b c f h w  -> b (f h w) c
        patch_out = rearrange(embs, "b c f h w -> b (f h w) c")

        return patch_out, grid_sizes

    def unpatchify(self, embs, frames, height, width):
        # b (f h w) (x y z c) -> b c (f x) (h y) (w z)
        patch_out = rearrange(
            embs,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=frames,
            h=height,
            w=width,
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )
        return patch_out

    def init_weights(self):
        if self.num_register_tokens > 0:
            nn.init.normal_(self.register_tokens, mean=0.0, std=0.02)
        for n, m in self.named_modules():
            if n == "":
                continue
            if hasattr(m, "reset_parameters"):
                # print(f"{n} -> reset_parameters")
                m.reset_parameters()
        

models = {
    "osp_next": OSPNextModel
}

models_main_block = {
    "osp_next": OSPNextAttentionBlock
}

models_blocks_to_float = {
    "osp_next": [OSPNextLayerNorm, OSPNextRMSNorm]
}

models_blocks_to_output_float = {
    "osp_next": None
}


if __name__ == "__main__":
    device = "cuda:0"
    dtype = torch.bfloat16
    model = OSPNextModel().to(device=device, dtype=dtype)
    model.set_gradient_checkpointing(True)
    x = torch.randn(2, 16, 21, 60, 104, device=device, dtype=dtype)
    t = torch.randint(0, 1000, (2,), device=device)
    context = torch.randn(2, 512, 4096, device=device, dtype=dtype)
    with torch.autocast("cuda", dtype=dtype):
        y = model(x, t, context)
    print(y.shape)