#!/bin/bash
wandb_dir="/home/ma-user/work/gyy/TorchDiff/output/osp_next_1_3b_81f480p_sparse2d2_ssp2_training_with_full/wandb"

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] syncing..."
    for run_dir in $wandb_dir/offline-run-*; do
        echo "syncing $run_dir"
        wandb sync --include-synced $run_dir
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] done, sleeping 30m..."
    sleep 30m
done