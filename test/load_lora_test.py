import torch

# 你的 LoRA 文件路径
lora_path = "/home/ma-user/work/xianyi/osp_next/TorchDiff/output/osp_next_14b_lr1e_5_81f720p_sparse2d2_sp2_ssp4_grpo4_node7_lr2e04/lora-checkpoint-20/adapter_model.bin"

# 加载模型权重字典 (State Dict)，映射到 CPU
state_dict = torch.load(lora_path, map_location="cpu", weights_only=True)

print(f"总计包含 {len(state_dict.keys())} 个张量 (Tensors)\n")
print("="*50)

# 遍历打印
for key, tensor in state_dict.items():
    print(f"张量名称 (Key): {key}")
    print(f"张量维度 (Shape): {tensor.shape}")
    print(f"张量数据类型 (Dtype): {tensor.dtype}")
    print(f"具体数值 (Values):\n{tensor}\n")
    print("-" * 50)
    
    # 如果你只想看前几个，可以在这里加上 break
    # break