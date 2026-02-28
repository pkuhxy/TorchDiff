import torch
import logging
from collections import OrderedDict
from typing import Hashable, Any
from accelerate.utils import is_npu_available as accelerate_is_npu_available

def str_to_precision(s):
    if s == "bfloat16" or s == "bf16":
        return torch.bfloat16
    elif s == "float16" or s == "fp16":
        return torch.float16
    elif s == "float32" or s == "float" or s == "fp32":
        return torch.float32
    elif s == "float64" or s == "double" or s == "fp64":
        return torch.float64
    elif s == "int64":
        return torch.int64
    elif s == "int32" or s == "int":
        return torch.int32
    elif s == "uint8":
        return torch.uint8
    else:
        raise ValueError(f"Unsupported precision string: {s}")

def precision_to_str(precision):
    if precision == torch.bfloat16:
        return "bfloat16"
    elif precision == torch.float16:
        return "float16"
    elif precision == torch.float32:
        return "float32"
    elif precision == torch.float64:
        return "float64"
    elif precision == torch.int64:
        return "int64"
    elif precision == torch.int32:
        return "int32"
    elif precision == torch.uint8:
        return "uint8"
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    
def params_nums_to_str(params_num):
    if params_num >= 1e9:
        return f"{params_num / 1e9:.2f}B"
    elif params_num >= 1e6:
        return f"{params_num / 1e6:.2f}M"
    elif params_num >= 1e3:
        return f"{params_num / 1e3:.2f}K"
    else:
        return str(params_num)

def get_memory_allocated():
    return f"{torch.cuda.memory_allocated() / 1024**3:.2f}"  # GiB

def is_npu_available():
    return accelerate_is_npu_available(True)

def check_and_import_npu():
    if is_npu_available():
        import torch_npu
        from torch_npu.contrib import transfer_to_npu
        torch_npu.npu.config.allow_internal_format = False

def safe_get_rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0

def contiguous(x):
    return x.contiguous() if not x.is_contiguous() else x

class SafeCacheManager:
    """带LRU淘汰机制的缓存管理器"""

    DEFAULT_MAX_CACHE_SIZE = 1

    def __init__(self, max_cache_size: int = DEFAULT_MAX_CACHE_SIZE):
        self.max_cache_size = max_cache_size
        self.cache: OrderedDict[Hashable, Any] = OrderedDict()

    def is_exist(self, key: Hashable) -> bool:
        return key in self.cache

    def get(self, key: Hashable) -> Any:
        try:
            value = self.cache[key]
            self.cache.move_to_end(key, last=True)
            return value
        except KeyError:
            return None

    def _evict_if_needed(self):
        """根据当前 max_cache_size 淘汰多余条目"""
        while len(self.cache) > self.max_cache_size > 0:
            self.cache.popitem(last=False)

    def set(self, key: Hashable, value: Any):
        """
        LRU 语义的 set：
        - 如果 key 已存在，更新并移到队尾
        - 如果 key 不存在，插入；若超过 max_cache_size，弹出最旧的一个
        """
        self.cache[key] = value
        self.cache.move_to_end(key, last=True)
        self._evict_if_needed()

    def clear(self):
        """清空缓存"""
        self.cache.clear()