import json
import glob
import shutil
import os
from tqdm import tqdm
from einops import rearrange

from torchdiff.data.utils.utils import LMDBReader, LMDBWriter
from torchdiff.utils.infer_utils import load_prompts

vbench_eval_data_save_dir = "/home/ma-user/work/gyy/TorchDiff/eval"

prompt_txt = "/home/ma-user/work/gyy/VBench/prompts/augmented_prompts/Wan2.1-T2V-1.3B/all_dimension_aug_wanx_seed42.txt"
prompts = load_prompts(prompt_txt)

with open(os.path.join(vbench_eval_data_save_dir, 'idx2prompt.txt'), 'w') as f:
    for idx, prompt in enumerate(prompts):
        f.write(f"{idx}\t{prompt}\n")

prompts = [{"cap": item} for item in prompts]

writer = LMDBWriter()
writer.save_filtered_data_samples(prompts, vbench_eval_data_save_dir)
test_reader = LMDBReader(vbench_eval_data_save_dir) 
print(test_reader.getitem(0))