import torch
from torchdiff.modules.osp_next import OSPNextModel

orig_weights_path = 'Wan2.1-T2V-14B/want2v_14b.pt'
save_path = 'output/osp_next_14b_init'

config = {
  'dim': 5120,
  'ffn_dim': 13824,
  'freq_dim': 256,
  'in_dim': 16,
  'num_heads': 40,
  'num_layers': 40,
  'out_dim': 16,
  'text_len': 512,
  'skiparse_model_type': "dual_end",
  'sparse_ratio': 2,
  'num_full_blocks': 8,
  'num_register_tokens': 0,
  'skiparse_1d': False,
  'skiparse_2d': True,
}

state_dict = torch.load(orig_weights_path, map_location='cpu')
model = OSPNextModel(**config)
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print(f"missing_keys: {missing_keys} \nunexpected_keys: {unexpected_keys}")
model.save_pretrained(save_path)