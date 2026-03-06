import torch
from torchdiff.utils.utils import is_npu_available

if not is_npu_available():
    torch._dynamo.config.cache_size_limit = 32
    torch._dynamo.config.accumulated_cache_size_limit = 32

# 如果用了cache就要disable编译，因为cache的key会变化
def maybe_compile(disable=False):
    def decorator(func):
        if is_npu_available():
            # NPU 上始终不编译
            return func
        if disable:
            # 非 NPU，但明确要求 disable → 禁止编译（包括被外层 compile 影响）
            return torch.compiler.disable(func)
        # 非 NPU，正常编译
        return torch.compile(func)
    return decorator