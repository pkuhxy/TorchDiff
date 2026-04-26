import torch
import torch_npu  # 关键：在 NPU 环境下必须引入此扩展包，否则 PyTorch 无法识别 'npu' 设备

# 你的 .pt 文件路径
model_path = "/home/ma-user/work/xianyi/osp_next/TorchDiff/output/osp_next_14b_lr1e_5_81f720p_sparse2d2_sp2_ssp4_grpo4_node8/iter_000000044/model_state_dict.pt"

print(f"正在加载模型权重: {model_path} ...")

# 设定 NPU 设备（如果你有多张卡，可以通过 "npu:0", "npu:1" 等来指定）
device = torch.device("npu:0")

# 加载权重
# 💡 map_location="npu:0" 会将权重直接加载到指定 NPU 的显存中
# 注意：如果文件特别大（比如 14B 模型），直接加载到一张卡可能会导致 NPU OOM (显存溢出)
# 如果只是为了查看数值，也可以保持 map_location="cpu" 先加载到内存，用 pdb 看完再决定
data = torch.load(model_path, map_location=device, weights_only=True)

import pdb; pdb.set_trace()

# 判断加载进来的是 state_dict (字典) 还是 包含了其他信息的对象
if isinstance(data, dict):
    print(f"成功加载！模型共包含 {len(data.keys())} 个张量 (Tensors)。\n")
    print("=" * 50)
    
    # 遍历并打印张量信息
    for key, tensor in data.items():
        print(f"张量名称 (Key): {key}")
        
        # 确保它是一个标准的 PyTorch 张量
        if isinstance(tensor, torch.Tensor):
            print(f"张量维度 (Shape): {tensor.shape}")
            print(f"数据类型 (Dtype): {tensor.dtype}")
            print(f"所在设备 (Device): {tensor.device}") # 会显示 npu:0
            print(f"数值预览:\n{tensor}\n")
        else:
            print(f"内容不是 Tensor，而是: {type(tensor)}")
            
        print("-" * 50)
        
        # 建议保留 break，否则张量太多会刷屏。如果想看全部，注释掉 break 即可
        break 
else:
    # 极少情况下，.pt 保存的不仅是权重，而是整个模型对象或其他结构
    print("加载的不是一个标准的字典 (state_dict)。")
    print(f"数据类型为: {type(data)}")
    print(data)