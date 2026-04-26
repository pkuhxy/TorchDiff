#!/bin/bash
wandb_dir="/home/ma-user/work/xianyi/osp_next/TorchDiff/output/osp_next_14b_lr1e_5_81f720p_sparse2d2_sp2_ssp4_grpo4_node7_lr1e04_hif8/wandb"



while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] syncing..."
    for run_dir in $wandb_dir/offline-run-*; do
        echo "syncing $run_dir"
        wandb sync --include-synced $run_dir
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] done, sleeping 30m..."
    sleep 30m
done