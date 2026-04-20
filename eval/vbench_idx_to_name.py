import os
import glob
import shutil

sample_videos_dir = "samples/vbench/osp_next_14b_81f720p_sparse2d2_ssp4_seed666_arg"
videos = glob.glob(f"{sample_videos_dir}/*.mp4")

idx_to_name_txt = "eval/idx2prompt.txt"
save_dir = "samples/vbench_renamed/osp_next_14b_81f720p_sparse2d2_ssp4_seed666_arg"
os.makedirs(save_dir, exist_ok=True)
with open(idx_to_name_txt, "r") as f:
    idx_to_name = {int(line.split('\t')[0]): line.split('\t')[1].strip() for line in f}

for video in videos:
    video_name = os.path.basename(video)
    idx = int(video_name.split("_")[1])
    local_idx = int(video_name.split("_")[-1].removesuffix(".mp4"))
    if idx in idx_to_name:
        new_name = idx_to_name[idx] + f"-{local_idx}.mp4"
        new_path = shutil.copy(video, os.path.join(save_dir, new_name))
        print(f"Renaming {video} to {new_path}")
        # Uncomment the next line to actually rename the files
        # os.rename(video, new_path)
    else:
        print(f"Index {idx} not found in idx_to_name mapping.")