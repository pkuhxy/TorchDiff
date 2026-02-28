"""
不通过wrapper方式实现的cp，用于更复杂的cp场景。
"""

from torch.distributed import ProcessGroup
import torch.distributed as dist

class ContextParallelState:
    global_rank: int = 0
    # 全局cp group，等价于skiparse cp和context cp的并集
    global_cp_group: ProcessGroup = None
    global_cp_rank: int = 0
    global_cp_size: int = 1
    # Ulysses context parallel group
    cp_group: ProcessGroup = None
    cp_rank: int = 0
    cp_size: int = 1
    # skiparse context parallel group
    skiparse_cp_group: ProcessGroup = None
    skiparse_cp_rank: int = 0
    skiparse_cp_size: int = 1
    # 是否初始化cp state
    is_initialized: bool = False

    def reset(self, global_cp_group: ProcessGroup = None, cp_group: ProcessGroup = None, skiparse_cp_group: ProcessGroup = None):
        self.global_rank = dist.get_rank() if dist.is_initialized() else 0
        if global_cp_group is not None:
            self.global_cp_group = global_cp_group
            self.global_cp_rank = dist.get_rank(global_cp_group)
            self.global_cp_size = dist.get_world_size(global_cp_group)
        if cp_group is not None:
            self.cp_group = cp_group
            self.cp_rank = dist.get_rank(cp_group)
            self.cp_size = dist.get_world_size(cp_group)
        if skiparse_cp_group is not None:
            self.skiparse_cp_group = skiparse_cp_group
            self.skiparse_cp_rank = dist.get_rank(skiparse_cp_group)
            self.skiparse_cp_size = dist.get_world_size(skiparse_cp_group)
        self.is_initialized = True

    def clear(self):
        self.global_rank = 0
        self.global_cp_group = None
        self.global_cp_rank = 0
        self.global_cp_size = 1
        self.cp_group = None
        self.cp_rank = 0
        self.cp_size = 1
        self.skiparse_cp_group = None
        self.skiparse_cp_rank = 0
        self.skiparse_cp_size = 1
        self.is_initialized = False

cp_state = ContextParallelState()

def use_context_parallel():
    return cp_state.is_initialized and cp_state.cp_size > 1

def use_skiparse_context_parallel():
    return cp_state.is_initialized and cp_state.skiparse_cp_size > 1