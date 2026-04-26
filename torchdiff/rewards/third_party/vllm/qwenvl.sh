#!/bin/bash
# ============================================================
# vLLM 启动脚本 - Qwen3-VL-8B-Instruct
# 用于 qwenvl_video_logit_score reward 的后端服务
# ============================================================

export VLLM_LOGGING_LEVEL=DEBUG

# export CUDA_VISIBLE_DEVICES=15

ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 vllm serve /home/ma-user/work/xianyi/ckpts/Qwen/Qwen3-VL-32B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --trust-remote-code \
  --enforce-eager \
