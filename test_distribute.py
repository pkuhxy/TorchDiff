import torch
import torch.nn as nn
from torch.distributed.tensor import distribute_tensor

p = nn.Parameter(torch.randn(10))
full_tensor = torch.randn(10)
try:
    device_mesh = p.device_mesh
except AttributeError:
    device_mesh = None

print("device_mesh:", device_mesh)
res = distribute_tensor(full_tensor, device_mesh, None)
print(res)
