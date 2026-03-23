"""
Test script for osp_sample_with_logprob()

用于调试 rollout 采样结果是否正确。
可以对比:
  1. SDE 采样 vs ODE (deterministic) 采样
  2. 纯 ODE Euler 去噪 (与推理 pipeline 完全一致) vs SDE / SDE-deterministic
  3. log_prob 值是否合理 (非 NaN, 非 Inf, 数值范围)
  4. latent 的统计信息 (mean, std, min, max)
  5. 最终解码出的视频是否正常

Usage (single GPU):
  torchrun --nproc_per_node=1 test/test_osp_sample_with_logprob.py --config <path_to_rl_config.yaml>

Usage (multi GPU, e.g. 8 GPUs with FSDP):
  torchrun --nproc_per_node=8 test/test_osp_sample_with_logprob.py --config <path_to_rl_config.yaml>

Optional flags:
  --num_inference_steps 20        覆盖配置中的步数
  --guidance_scale 5.0            覆盖 CFG scale
  --deterministic                 使用 ODE (无噪声) 采样
  --save_video                    保存采样视频到 output_dir
  --output_dir ./test_output      视频保存目录
  --prompt "a cat running"        自定义 prompt (否则用配置中的 prompt_file)
  --seed 42                       随机种子
  --skip_vae_decode               跳过 VAE decode，只测试 latent 采样
  --compare_deterministic         同时跑 SDE 和 ODE 做对比
  --compare_ode                   同时跑纯 ODE Euler 去噪 (与推理 pipeline 一致) 做对比
"""

import os
import sys
import math
import yaml
import time
import copy
import numpy as np
from argparse import ArgumentParser

import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from torchdiff.utils.utils import check_and_import_npu, is_npu_available
check_and_import_npu()

import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from torchdiff.modules import (
    WanVAE,
    T5EncoderModel,
    models,
    models_main_block,
    models_blocks_to_float,
    models_blocks_to_output_float,
)
from torchdiff.schedulers import schedulers
from torchdiff.distributed.utils import setup_distributed_env, cleanup_distributed_env
from torchdiff.distributed.fsdp2_wrapper import FSDP2_mix_wrapper
from torchdiff.distributed.cp_state import cp_state
from torchdiff.distributed.checkpoint import Checkpointer
from torchdiff.utils.utils import str_to_precision, get_memory_allocated
from torchdiff.utils.log_utils import get_logger, log_on_main_process
from torchdiff.utils.random_utils import set_seed
from torchdiff.data.utils.wan_utils import WanTextProcessor
from transformers import AutoTokenizer

# Import the functions we want to test
from train.train_osp_RL import sde_step_with_logprob, osp_sample_with_logprob


# ==================== 纯 ODE Euler 去噪 (与推理 pipeline 完全一致) ====================

@torch.no_grad()
def osp_sample_ode(
    model,
    scheduler,
    vae,
    latent_shape,
    text_embeddings,
    device,
    weight_dtype,
    num_inference_steps=50,
    guidance_scale=5.0,
    negative_text_embeddings=None,
    start_frame_latents=None,
    cp_group=None,
):
    """
    纯 ODE Euler 去噪，与推理 pipeline (FlowMatchingScheduler.sample + _step) 完全一致。

    步进公式: latents = latents + model_output * (sigma_{i+1} - sigma_i)

    这个函数不包含任何 SDE 噪声、drift 修正项，用于验证模型本身的去噪能力。
    如果 ODE 采样正确而 SDE 采样不正确，说明 sde_step_with_logprob 的 SDE 公式有问题。

    Returns:
        videos: decoded video tensor [B, C, T, H, W] in float, range [-1, 1]
        all_latents: list of latent tensors at each step (length = num_steps + 1)
    """
    B, C, T, H, W = latent_shape
    do_cfg = guidance_scale > 1.0

    # 1. 生成初始噪声 (与 osp_sample_with_logprob 完全一样)
    latents = torch.randn(latent_shape, device=device, dtype=torch.float32)

    if cp_group is not None:
        torch.distributed.broadcast(latents, src=dist.get_global_rank(cp_group, 0), group=cp_group)

    # 2. 构建 sigma schedule (与推理 pipeline _set_sigmas 一致)
    sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
    if hasattr(scheduler, 'shift') and scheduler.shift != 1.0:
        shift = scheduler.shift
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

    timesteps = sigmas * 1000.0

    all_latents = [latents.clone()]

    # 3. Euler ODE 去噪循环
    for i in range(num_inference_steps):
        torch.cuda.synchronize()

        latents_input = latents.to(weight_dtype)
        t = timesteps[i]
        t_batch = t.expand(B).to(device)

        # 条件预测
        with torch.autocast("cuda", dtype=weight_dtype):
            noise_pred = model(
                latents_input,
                t_batch,
                text_embeddings,
                start_frame_latents=start_frame_latents,
            )
        torch.cuda.synchronize()

        # CFG
        if do_cfg and negative_text_embeddings is not None:
            with torch.autocast("cuda", dtype=weight_dtype):
                noise_uncond = model(
                    latents_input,
                    t_batch,
                    negative_text_embeddings,
                    start_frame_latents=start_frame_latents,
                )
            torch.cuda.synchronize()
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
            del noise_uncond

        # ===== 核心: 纯 Euler ODE step =====
        # 与 FlowMatchingScheduler._step 完全一致:
        #   next_latents = latents + model_output * delta_t
        #   delta_t = sigmas[i+1] - sigmas[i]  (负值，因为 sigma 从 1 递减到 0)
        delta_t = sigmas[i + 1] - sigmas[i]
        latents = latents.float() + noise_pred.float() * delta_t.float()

        del noise_pred, latents_input

        all_latents.append(latents.clone())

        if (i + 1) % 5 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # 4. VAE decode
    all_latents_cpu = [l.cpu() for l in all_latents]
    del all_latents
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    with torch.no_grad():
        videos = vae.decode(latents)
    del latents

    return videos, all_latents_cpu


def print_tensor_stats(name, tensor, rank=0):
    """打印 tensor 的统计信息用于调试"""
    if dist.get_rank() != rank:
        return
    if tensor is None:
        print(f"  [{name}] None")
        return
    t = tensor.float()
    print(
        f"  [{name}] shape={list(tensor.shape)}, dtype={tensor.dtype}, "
        f"mean={t.mean().item():.6f}, std={t.std().item():.6f}, "
        f"min={t.min().item():.6f}, max={t.max().item():.6f}, "
        f"nan={torch.isnan(t).sum().item()}, inf={torch.isinf(t).sum().item()}"
    )


def test_sde_step_with_logprob(device, rank):
    """单元测试: sde_step_with_logprob 的数值行为"""
    if rank != 0:
        return

    print("\n" + "=" * 60)
    print("TEST 1: sde_step_with_logprob 数值测试")
    print("=" * 60)

    num_steps = 20
    B, C, T, H, W = 2, 16, 21, 60, 104
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    # 模拟一步 SDE step
    sample = torch.randn(B, C, T, H, W, device=device)
    model_output = torch.randn(B, C, T, H, W, device=device)

    for step_idx in [0, num_steps // 2, num_steps - 1]:
        prev_sample, log_prob, prev_mean, std_scale = sde_step_with_logprob(
            sigmas, model_output, step_idx, sample, num_steps,
        )
        print(f"\n  Step {step_idx}/{num_steps}:")
        print(f"    sigma={sigmas[step_idx].item():.4f} -> sigma_next={sigmas[step_idx + 1].item():.4f}")
        print_tensor_stats("prev_sample", prev_sample, rank)
        print_tensor_stats("log_prob", log_prob, rank)
        print_tensor_stats("prev_mean", prev_mean, rank)
        print_tensor_stats("std_scale", std_scale, rank)

        # 验证 log_prob 合理性
        assert not torch.isnan(log_prob).any(), f"log_prob contains NaN at step {step_idx}!"
        assert not torch.isinf(log_prob).any(), f"log_prob contains Inf at step {step_idx}!"

    # 测试 deterministic 模式
    prev_sample_det, log_prob_det, _, _ = sde_step_with_logprob(
        sigmas, model_output, 0, sample, num_steps, determistic=True,
    )
    print(f"\n  Deterministic step 0:")
    print_tensor_stats("prev_sample_det", prev_sample_det, rank)
    print_tensor_stats("log_prob_det", log_prob_det, rank)

    # 测试给定 prev_sample 的 log_prob 计算
    prev_sample_given = torch.randn(B, C, T, H, W, device=device)
    _, log_prob_given, _, _ = sde_step_with_logprob(
        sigmas, model_output, 0, sample, num_steps,
        prev_sample=prev_sample_given,
    )
    print(f"\n  With given prev_sample:")
    print_tensor_stats("log_prob_given", log_prob_given, rank)

    print("\n  ✅ sde_step_with_logprob 通过基本测试")


def test_sigma_schedule(scheduler, device, num_inference_steps, rank):
    """测试 sigma schedule 的构建"""
    if rank != 0:
        return

    print("\n" + "=" * 60)
    print("TEST 2: Sigma Schedule 测试")
    print("=" * 60)

    sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
    print(f"\n  原始 sigmas (first 5): {sigmas[:5].tolist()}")
    print(f"  原始 sigmas (last 5):  {sigmas[-5:].tolist()}")

    if hasattr(scheduler, 'shift') and scheduler.shift != 1.0:
        shift = scheduler.shift
        sigmas_shifted = shift * sigmas / (1 + (shift - 1) * sigmas)
        print(f"\n  Shift = {shift}")
        print(f"  Shifted sigmas (first 5): {sigmas_shifted[:5].tolist()}")
        print(f"  Shifted sigmas (last 5):  {sigmas_shifted[-5:].tolist()}")
        sigmas = sigmas_shifted

    timesteps = sigmas * 1000.0
    print(f"\n  Timesteps (first 5): {timesteps[:5].tolist()}")
    print(f"  Timesteps (last 5):  {timesteps[-5:].tolist()}")

    # 验证单调递减
    diffs = sigmas[1:] - sigmas[:-1]
    assert (diffs <= 0).all(), "Sigma schedule 不是单调递减的!"
    print("\n  ✅ Sigma schedule 单调递减")


def test_full_sampling(
    model, scheduler, vae, text_encoder,
    device, weight_dtype, config, args, rank,
):
    """完整测试: osp_sample_with_logprob 采样"""
    print("\n" + "=" * 60)
    print("TEST 3: osp_sample_with_logprob 完整采样测试")
    print("=" * 60)

    rl_config = config.get("rl_config", {})
    num_inference_steps = args.num_inference_steps or rl_config.get("num_inference_steps", 30)
    guidance_scale = args.guidance_scale or rl_config.get("guidance_scale", 5.0)
    video_height = rl_config.get("height", 480)
    video_width = rl_config.get("width", 832)
    video_num_frames = rl_config.get("num_frames", 49)
    model_config = config.get("model_config", {})
    text_encoder_config = config.get("text_encoder_config", {})
    sample_batch_size = 1  # 测试用 batch=1

    # Latent shape
    latent_T = (video_num_frames - 1) // 4 + 1  # VAE temporal factor = 4
    latent_H = video_height // 8                  # VAE spatial factor = 8
    latent_W = video_width // 8
    latent_C = model_config.get("in_dim", 16)
    latent_shape = (sample_batch_size, latent_C, latent_T, latent_H, latent_W)

    if rank == 0:
        print(f"\n  Video: {video_num_frames}f x {video_height}h x {video_width}w")
        print(f"  Latent shape: {latent_shape}")
        print(f"  Num inference steps: {num_inference_steps}")
        print(f"  Guidance scale: {guidance_scale}")
        print(f"  Weight dtype: {weight_dtype}")
        print(f"  Deterministic: {args.deterministic}")

    # Prepare text embeddings
    # if args.prompt:
    #     prompts = [args.prompt]
    # else:
    #     prompt_file = rl_config.get("prompt_file", None)
    #     if prompt_file and os.path.exists(prompt_file):
    #         with open(prompt_file, 'r') as f:
    #             prompts = [line.strip() for line in f if line.strip()][:1]
    #     else:
    prompts = ["A beautiful sunset over the ocean with waves crashing on the shore."]

    if rank == 0:
        print(f"  Prompt: {prompts[0][:100]}...")

    # Tokenize prompt
    data_config = config.get("data_config", {})
    text_tokenizer_path = data_config.get("dataset_config", {}).get("text_tokenizer_path", None)
    text_max_length = data_config.get("dataset_config", {}).get(
        "tokenizer_max_length",
        text_encoder_config.get("text_len", 512)
    )

    if text_tokenizer_path is None:
        # 尝试从 text_encoder_config 中提取
        text_tokenizer_path = text_encoder_config.get("text_tokenizer_path", None)

    if text_tokenizer_path is None:
        raise ValueError(
            "无法找到 text_tokenizer_path，请在配置文件中设置 "
            "data_config.dataset_config.text_tokenizer_path"
        )

    text_processor = WanTextProcessor(
        tokenizer=AutoTokenizer.from_pretrained(text_tokenizer_path),
        model_max_length=text_max_length,
        return_prompt_mask=True,
    )

    prompt_ids, prompt_mask = text_processor(prompts[0])
    prompt_ids = prompt_ids.to(device)
    prompt_mask = prompt_mask.to(device)

    with torch.no_grad():
        text_embeddings = text_encoder(prompt_ids, prompt_mask)
    print_tensor_stats("text_embeddings", text_embeddings, rank)

    # Negative text embeddings for CFG
    # 与推理 pipeline 保持一致：使用真实的 negative prompt 文本而非全零 token
    from torchdiff.utils.constant import NEGATIVE_PROMOPT
    neg_prompt_ids, neg_prompt_mask = text_processor(NEGATIVE_PROMOPT)
    neg_prompt_ids = neg_prompt_ids.to(device)
    neg_prompt_mask = neg_prompt_mask.to(device)
    with torch.no_grad():
        neg_text_embeddings = text_encoder(neg_prompt_ids, neg_prompt_mask)
    print_tensor_stats("neg_text_embeddings", neg_text_embeddings, rank)

    # ===================== SDE 采样 =====================
    if rank == 0:
        print(f"\n{'─' * 40}")
        print("  开始 SDE 采样 (deterministic={})...".format(args.deterministic))

    set_seed(args.seed, device_specific=False)
    torch.cuda.synchronize()
    t_start = time.time()

    model.eval()
    with torch.no_grad():
        videos, all_latents, all_log_probs, all_kl = osp_sample_with_logprob(
            model=model,
            scheduler=scheduler,
            vae=vae if not args.skip_vae_decode else _DummyVAE(device),
            latent_shape=latent_shape,
            text_embeddings=text_embeddings,
            device=device,
            weight_dtype=weight_dtype,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_text_embeddings=neg_text_embeddings.expand(sample_batch_size, -1, -1),
            start_frame_latents=None,
            determistic=args.deterministic,
            kl_reward=0.0,
            ref_model=None,
            cp_group=None,
        )

    torch.cuda.synchronize()
    t_elapsed = time.time() - t_start

    if rank == 0:
        print(f"\n  采样耗时: {t_elapsed:.2f}s")
        print(f"\n  === 采样结果统计 ===")
        print(f"  all_latents 长度: {len(all_latents)} (应为 num_steps+1 = {num_inference_steps + 1})")
        print(f"  all_log_probs 长度: {len(all_log_probs)} (应为 num_steps = {num_inference_steps})")
        print(f"  all_kl 长度: {len(all_kl)} (应为 num_steps = {num_inference_steps})")

        # 检查 latent 维度
        assert len(all_latents) == num_inference_steps + 1, \
            f"all_latents 长度错误: {len(all_latents)} != {num_inference_steps + 1}"
        assert len(all_log_probs) == num_inference_steps, \
            f"all_log_probs 长度错误: {len(all_log_probs)} != {num_inference_steps}"

        # 打印每一步的 latent & log_prob 统计
        print(f"\n  === 逐步 Latent 统计 ===")
        for i in range(0, len(all_latents), max(1, len(all_latents) // 10)):
            print_tensor_stats(f"latent[{i}]", all_latents[i], rank)
        print(f"  ...")
        print_tensor_stats(f"latent[{len(all_latents)-1}] (final)", all_latents[-1], rank)

        print(f"\n  === 逐步 Log Prob 统计 ===")
        lp_values = []
        for i in range(len(all_log_probs)):
            lp = all_log_probs[i]
            lp_values.append(lp.mean().item())
            if i % max(1, len(all_log_probs) // 10) == 0:
                print_tensor_stats(f"log_prob[{i}]", lp, rank)
        print(f"  ...")
        print_tensor_stats(f"log_prob[{len(all_log_probs)-1}] (final)", all_log_probs[-1], rank)

        # log_prob 的总体统计
        all_lp = torch.stack(all_log_probs)
        print(f"\n  === Log Prob 总体统计 ===")
        print(f"    Overall mean: {all_lp.mean().item():.6f}")
        print(f"    Overall std:  {all_lp.std().item():.6f}")
        print(f"    Per-step mean: {[f'{v:.4f}' for v in lp_values[:5]]} ... {[f'{v:.4f}' for v in lp_values[-5:]]}")
        print(f"    Sum (total log_prob): {all_lp.sum(dim=0).mean().item():.6f}")
        print(f"    NaN count: {torch.isnan(all_lp).sum().item()}")
        print(f"    Inf count: {torch.isinf(all_lp).sum().item()}")

        # 检查 log_prob 合理性
        if torch.isnan(all_lp).any():
            print("  ⚠️ WARNING: log_prob 包含 NaN!")
        if torch.isinf(all_lp).any():
            print("  ⚠️ WARNING: log_prob 包含 Inf!")
        if all_lp.abs().max() > 1e6:
            print(f"  ⚠️ WARNING: log_prob 数值过大 (max abs = {all_lp.abs().max().item():.2f})")

        # KL 统计
        print(f"\n  === KL 统计 ===")
        all_kl_t = torch.stack(all_kl)
        print(f"    KL mean: {all_kl_t.mean().item():.6f}")
        print(f"    KL std:  {all_kl_t.std().item():.6f}")

        # Videos 统计
        if not args.skip_vae_decode:
            print(f"\n  === Video 统计 ===")
            print_tensor_stats("videos", videos, rank)
            vid_range = (videos.min().item(), videos.max().item())
            print(f"    Value range: [{vid_range[0]:.4f}, {vid_range[1]:.4f}] (期望 [-1, 1])")
            if vid_range[0] < -1.01 or vid_range[1] > 1.01:
                print("  ⚠️ WARNING: video 值超出 [-1, 1] 范围!")

    # ===================== 对比 Deterministic 采样 (SDE with determistic=True) =====================
    if args.compare_deterministic and not args.deterministic:
        if rank == 0:
            print(f"\n{'─' * 40}")
            print("  开始 SDE-deterministic (determistic=True) 采样做对比...")
            print("  注意: 这是 sde_step_with_logprob 的 determistic 模式，")
            print("        公式为 prev_sample = sample + dt * model_output (覆盖 SDE drift+noise)")

        set_seed(args.seed, device_specific=False)
        with torch.no_grad():
            videos_det, all_latents_det, all_log_probs_det, all_kl_det = osp_sample_with_logprob(
                model=model,
                scheduler=scheduler,
                vae=vae if not args.skip_vae_decode else _DummyVAE(device),
                latent_shape=latent_shape,
                text_embeddings=text_embeddings,
                device=device,
                weight_dtype=weight_dtype,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_text_embeddings=neg_text_embeddings.expand(sample_batch_size, -1, -1),
                start_frame_latents=None,
                determistic=True,
                kl_reward=0.0,
                ref_model=None,
                cp_group=None,
            )

        if rank == 0:
            print(f"\n  === SDE vs SDE-deterministic 对比 ===")
            # 初始 latent 应该一样（相同 seed）
            latent_0_diff = (all_latents[0] - all_latents_det[0]).abs().max().item()
            print(f"    初始 latent 差异 (max abs): {latent_0_diff:.8f} (应为 0)")

            # 最终 latent 差异
            latent_final_diff = (all_latents[-1] - all_latents_det[-1]).abs().mean().item()
            print(f"    最终 latent 差异 (mean abs): {latent_final_diff:.6f}")

            # log_prob 差异
            all_lp_det = torch.stack(all_log_probs_det)
            print(f"\n    SDE log_prob sum mean:           {torch.stack(all_log_probs).sum(dim=0).mean().item():.6f}")
            print(f"    SDE-deterministic log_prob sum:   {all_lp_det.sum(dim=0).mean().item():.6f}")

            # Video 统计
            if not args.skip_vae_decode:
                print_tensor_stats("SDE-deterministic video", videos_det, rank)
                vid_range_det = (videos_det.min().item(), videos_det.max().item())
                print(f"    SDE-deterministic video range: [{vid_range_det[0]:.4f}, {vid_range_det[1]:.4f}]")
                video_diff = (videos - videos_det).abs().mean().item()
                print(f"    SDE vs SDE-deterministic video diff (mean abs): {video_diff:.6f}")

    # ===================== 纯 ODE Euler 对比 (与推理 pipeline 完全一致) =====================
    videos_pure_ode = None
    all_latents_pure_ode = None
    if args.compare_ode:
        if rank == 0:
            print(f"\n{'─' * 40}")
            print("  开始纯 ODE Euler 去噪 (与推理 pipeline FlowMatchingScheduler._step 完全一致)")
            print("  公式: latents = latents + model_output * (sigma_{i+1} - sigma_i)")
            print("  没有 SDE drift 修正项，没有高斯噪声")

        set_seed(args.seed, device_specific=False)
        videos_pure_ode, all_latents_pure_ode = osp_sample_ode(
            model=model,
            scheduler=scheduler,
            vae=vae if not args.skip_vae_decode else _DummyVAE(device),
            latent_shape=latent_shape,
            text_embeddings=text_embeddings,
            device=device,
            weight_dtype=weight_dtype,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_text_embeddings=neg_text_embeddings.expand(sample_batch_size, -1, -1),
            start_frame_latents=None,
            cp_group=None,
        )

        if rank == 0:
            print(f"\n  === 纯 ODE Euler 结果 ===")
            print_tensor_stats("ODE final latent", all_latents_pure_ode[-1], rank)

            if not args.skip_vae_decode:
                print_tensor_stats("ODE video", videos_pure_ode, rank)
                vid_range_ode = (videos_pure_ode.min().item(), videos_pure_ode.max().item())
                print(f"    ODE video range: [{vid_range_ode[0]:.4f}, {vid_range_ode[1]:.4f}] (期望 [-1, 1])")

            # 逐步对比 latent
            print(f"\n  === SDE vs 纯 ODE 逐步 latent 对比 ===")
            latent_0_diff = (all_latents[0] - all_latents_pure_ode[0]).abs().max().item()
            print(f"    Step 0  初始 latent 差异 (max abs): {latent_0_diff:.8f} (应为 0)")
            for check_step in [1, 2, 5, 10, num_inference_steps // 2, num_inference_steps]:
                if check_step < len(all_latents) and check_step < len(all_latents_pure_ode):
                    diff_mean = (all_latents[check_step] - all_latents_pure_ode[check_step]).abs().mean().item()
                    diff_max = (all_latents[check_step] - all_latents_pure_ode[check_step]).abs().max().item()
                    print(f"    Step {check_step:3d} latent 差异 — mean abs: {diff_mean:.6f}, max abs: {diff_max:.6f}")

            latent_final_diff = (all_latents[-1] - all_latents_pure_ode[-1]).abs().mean().item()
            print(f"\n    最终 latent 差异 (mean abs): {latent_final_diff:.6f}")

            if not args.skip_vae_decode:
                video_diff = (videos - videos_pure_ode).abs().mean().item()
                video_diff_max = (videos - videos_pure_ode).abs().max().item()
                print(f"    SDE vs ODE video diff — mean abs: {video_diff:.6f}, max abs: {video_diff_max:.6f}")

            # 如果也跑了 SDE-deterministic，三方对比
            if args.compare_deterministic and not args.deterministic:
                print(f"\n  === 三方对比: SDE vs SDE-deterministic vs 纯 ODE ===")
                det_vs_ode_diff = (all_latents_det[-1] - all_latents_pure_ode[-1]).abs().mean().item()
                print(f"    SDE-deterministic vs 纯 ODE final latent diff (mean abs): {det_vs_ode_diff:.8f}")
                print(f"    (如果 > 0，说明 sde_step_with_logprob(determistic=True) 和 Euler ODE 步进有差异!)")

                if not args.skip_vae_decode:
                    det_vs_ode_video_diff = (videos_det - videos_pure_ode).abs().mean().item()
                    print(f"    SDE-deterministic vs 纯 ODE video diff (mean abs): {det_vs_ode_video_diff:.6f}")

    # ===================== 保存视频 =====================
    if args.save_video and rank == 0:
        import imageio
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        def _save_video(video_tensor, label):
            for b in range(video_tensor.shape[0]):
                video_np = video_tensor[b].float().cpu().numpy().transpose(1, 2, 3, 0)  # T, H, W, C
                frames = [((frame + 1) / 2 * 255).clip(0, 255).astype(np.uint8) for frame in video_np]
                path = os.path.join(output_dir, f"test_sample_{label}_{b}.mp4")
                imageio.mimsave(path, frames, fps=16, codec="libx264", format='FFMPEG')
                print(f"  保存视频: {path}")

        if not args.skip_vae_decode:
            _save_video(videos, "sde")

            if args.compare_deterministic and not args.deterministic:
                _save_video(videos_det, "sde_deterministic")

            if args.compare_ode and videos_pure_ode is not None:
                _save_video(videos_pure_ode, "pure_ode")

    # ===================== 测试 log_prob 一致性 =====================
    if rank == 0:
        print(f"\n{'─' * 40}")
        print("  TEST 4: Log prob 一致性验证")
        print("  (用采样时记录的 latent 重新计算 log_prob, 验证是否一致)")

    # 重新构建 sigma schedule
    sigmas_schedule = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
    if hasattr(scheduler, 'shift') and scheduler.shift != 1.0:
        shift = scheduler.shift
        sigmas_schedule = shift * sigmas_schedule / (1 + (shift - 1) * sigmas_schedule)

    # 选几个步骤验证: 用记录的 latent[i] 和 latent[i+1] 计算 log_prob
    # 然后和 all_log_probs[i] 对比
    # 注意: 需要重新跑 model forward 来获取 noise_pred, 这里只验证 sde_step_with_logprob 的一致性
    if rank == 0 and not args.deterministic:
        check_steps = [0, num_inference_steps // 4, num_inference_steps // 2, num_inference_steps - 1]
        for step_idx in check_steps:
            if step_idx >= len(all_log_probs):
                continue
            latent_cur = all_latents[step_idx].to(device)
            latent_next = all_latents[step_idx + 1].to(device)

            # 我们不能在这里重新跑 model forward（因为可能已经 moved parameters）
            # 但可以验证: 给定 prev_sample=latent_next, 重新调用 sde_step_with_logprob
            # 如果 model_output 一样，log_prob 应该一样
            # 这里只是验证给定 prev_sample 时的 log_prob 计算是自洽的
            pass

        print("  (log_prob 一致性需要重新 forward，此处跳过 — 请查看上方的 log_prob 统计)")

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("所有测试完成!")
        print(f"{'=' * 60}")
        print(f"\n  总结:")
        print(f"  - 如果纯 ODE 视频正确而 SDE 视频不正确 → sde_step_with_logprob 的 SDE 公式有问题")
        print(f"  - 如果 SDE-deterministic 和纯 ODE 结果不同 → sde_step_with_logprob(determistic=True)")
        print(f"    的实现与推理 pipeline 的 Euler step 不一致")
        print(f"  - 如果纯 ODE 视频也不正确 → 问题在模型本身 / sigma schedule / CFG 等")
        print(f"{'=' * 60}\n")


class _DummyVAE:
    """跳过 VAE decode 时的替代品"""
    def __init__(self, device):
        self.device = device

    def decode(self, latents):
        B, C, T, H, W = latents.shape
        # 返回假视频: [B, 3, T*4, H*8, W*8]，但这里简化为返回 latents 本身
        return torch.zeros(B, 3, (T - 1) * 4 + 1, H * 8, W * 8, device=latents.device)


def main():
    parser = ArgumentParser(description="Test osp_sample_with_logprob()")
    parser.add_argument("--config", type=str, required=True, help="Path to RL training config YAML")
    parser.add_argument("--num_inference_steps", type=int, default=None, help="Override num_inference_steps")
    parser.add_argument("--guidance_scale", type=float, default=None, help="Override guidance_scale")
    parser.add_argument("--deterministic", action="store_true", help="Use ODE (deterministic) sampling")
    parser.add_argument("--save_video", action="store_true", help="Save sampled videos")
    parser.add_argument("--output_dir", type=str, default="./test_sample_output", help="Output directory")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt text")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip_vae_decode", action="store_true", help="Skip VAE decode (only test latent sampling)")
    parser.add_argument("--compare_deterministic", action="store_true", help="Also run deterministic sampling for comparison")
    parser.add_argument("--compare_ode", action="store_true", help="Also run pure ODE Euler denoising (same as inference pipeline) for comparison")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise ValueError(f"Config file {args.config} does not exist!")

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    logger = get_logger()

    # ========== Distributed Setup ==========
    setup_distributed_env()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    weight_dtype = str_to_precision(config.get("weight_dtype", "bfloat16"))

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"  osp_sample_with_logprob 测试脚本")
        print(f"  World size: {world_size}, Rank: {rank}")
        print(f"  Device: {device}, Weight dtype: {weight_dtype}")
        print(f"{'=' * 60}")

    # ========== FSDP Mesh ==========
    fsdp_size = config.get("fsdp_size", world_size)
    if fsdp_size > world_size:
        fsdp_size = world_size
    ddp_size = config.get("ddp_size", world_size // fsdp_size)
    ddp_fsdp_mesh = init_device_mesh("cuda", (ddp_size, fsdp_size), mesh_dim_names=("ddp", "fsdp"))

    set_seed(args.seed, device_specific=False)

    # ========== Init Models ==========
    model_name = config.get("model_name", "osp_next")
    model_config = config.get("model_config", {})
    vae_config = config.get("vae_config", {})
    text_encoder_config = config.get("text_encoder_config", {})
    scheduler_config = config.get("scheduler_config", {})

    # VAE
    log_on_main_process(logger, "Initializing VAE...")
    vae = WanVAE(
        vae_pth=vae_config.get("vae_path", None),
        dtype=str_to_precision(vae_config.get("dtype", "fp32")),
        device=device,
    )
    log_on_main_process(logger, f"VAE initialized, memory: {get_memory_allocated()} GiB")

    # Text Encoder
    log_on_main_process(logger, "Initializing text encoder...")
    text_encoder = T5EncoderModel(
        text_len=text_encoder_config.get("text_len", 512),
        dtype=text_encoder_config.get("dtype", weight_dtype),
        device=device,
        checkpoint_path=text_encoder_config.get("checkpoint_path", None),
        use_fsdp=text_encoder_config.get("use_fsdp", False),
        device_mesh=ddp_fsdp_mesh if text_encoder_config.get("use_fsdp", False) else None,
    )
    log_on_main_process(logger, f"Text encoder initialized, memory: {get_memory_allocated()} GiB")

    # Scheduler
    log_on_main_process(logger, "Initializing scheduler...")
    scheduler_config_copy = copy.deepcopy(scheduler_config)
    scheduler_name = scheduler_config_copy.pop("scheduler_name", "flow_matching")
    scheduler = schedulers[scheduler_name](**scheduler_config_copy)
    log_on_main_process(logger, f"Scheduler: {scheduler_name}, shift={getattr(scheduler, 'shift', 'N/A')}")

    # Model
    log_on_main_process(logger, "Initializing diffusion model...")
    pretrained_model_dir_or_checkpoint = model_config.get("pretrained_model_dir_or_checkpoint", None)
    has_loaded_pretrained_model = False
    if pretrained_model_dir_or_checkpoint is not None and os.path.isdir(pretrained_model_dir_or_checkpoint):
        log_on_main_process(logger, f"Loading from dir: {pretrained_model_dir_or_checkpoint}")
        model = models[model_name].from_pretrained(pretrained_model_dir_or_checkpoint)
        has_loaded_pretrained_model = True
    else:
        log_on_main_process(logger, "Initializing model from scratch (will load weights later)")
        with torch.device("meta"):
            model = models[model_name](**model_config)

    model.eval()

    # FSDP2 wrap
    FSDP2_mix_wrapper(
        model,
        dp_mesh=ddp_fsdp_mesh,
        weight_dtype=weight_dtype,
        main_block_to_half=models_main_block[model_name],
        blocks_to_float=models_blocks_to_float[model_name],
        blocks_to_output_float=models_blocks_to_output_float[model_name],
        reshard_after_forward=True,  # 测试时用 True 节省显存
        cpu_offload=config.get("model_cpu_offload", False),
    )

    if not has_loaded_pretrained_model:
        model.to_empty(device=device)
        set_seed(args.seed, device_specific=False)
        model.reset_parameters()

    # Load checkpoint
    if pretrained_model_dir_or_checkpoint is not None and os.path.isfile(pretrained_model_dir_or_checkpoint):
        log_on_main_process(logger, f"Loading weights from: {pretrained_model_dir_or_checkpoint}")
        save_with_dcp_api = config.get("save_with_dcp_api", False)
        Checkpointer.load_model_from_path(model, pretrained_model_dir_or_checkpoint, dcp_api=save_with_dcp_api)

    # Also check for training checkpoint
    output_dir = config.get("output_dir", "./output_rl")
    if os.path.isdir(output_dir):
        save_with_dcp_api = config.get("save_with_dcp_api", False)
        checkpointer = Checkpointer(folder=output_dir, dcp_api=save_with_dcp_api)
        if checkpointer.last_training_iteration is not None:
            log_on_main_process(logger, f"Loading training checkpoint iter {checkpointer.last_training_iteration}...")
            checkpointer.load_model(model)
            has_loaded_pretrained_model = True

    log_on_main_process(logger, f"Model initialized, memory: {get_memory_allocated()} GiB")

    # ========== Run Tests ==========
    # Test 1: sde_step_with_logprob 单元测试
    test_sde_step_with_logprob(device, rank)

    # Test 2: sigma schedule 测试
    rl_config = config.get("rl_config", {})
    num_steps = args.num_inference_steps or rl_config.get("num_inference_steps", 20)
    test_sigma_schedule(scheduler, device, num_steps, rank)

    # Test 3 & 4: 完整采样测试
    test_full_sampling(
        model=model,
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        device=device,
        weight_dtype=weight_dtype,
        config=config,
        args=args,
        rank=rank,
    )

    cleanup_distributed_env()


if __name__ == "__main__":
    main()
