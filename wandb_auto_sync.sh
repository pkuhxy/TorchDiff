#!/bin/bash
wandb_dir="/home/ma-user/work/gyy/TorchDiff/output/hif8/osp_next_14b_81f720p_sparse2d2_ssp4_from_full_seed1024/wandb"

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] syncing..."
    for run_dir in $wandb_dir/offline-run-*; do
        echo "syncing $run_dir"
        wandb sync --include-synced $run_dir
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] done, sleeping 30m..."
    sleep 30m
done