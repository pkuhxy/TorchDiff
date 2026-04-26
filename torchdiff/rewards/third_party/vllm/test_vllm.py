#!/usr/bin/env python3
"""
独立测试脚本: qwenvl_video_logit_score

测试 QwenVL Video/Image reward 函数，通过 vLLM 服务获取 logit 评分。

使用前需要先启动 vLLM 服务：
    python -m vllm.entrypoints.openai.api_server \
        --model /apdcephfs_nj7/share_1220751/xianyihe/ckpts/Qwen/Qwen3-VL-8B-Instruct \
        --tensor-parallel-size 1 \
        --port 8000 \
        --trust-remote-code

用法:
    # 测试单张图片
    python test_qwenvl_video_logit_score.py --image path/to/image.jpg --prompt "a cat sitting on a table"

    # 测试多张图片
    python test_qwenvl_video_logit_score.py --image img1.jpg img2.jpg --prompt "a cat" "a dog"

    # 测试视频（输入多帧图片作为视频）
    python test_qwenvl_video_logit_score.py --video_frames frame1.jpg frame2.jpg frame3.jpg --prompt "a person walking"

    # 测试随机 tensor 输入（不需要实际图片文件）
    python test_qwenvl_video_logit_score.py --test_random_image --prompt "a beautiful landscape"
    python test_qwenvl_video_logit_score.py --test_random_video --prompt "a person dancing" --num_frames 8

    # 自定义 vLLM 服务地址和模型名
    python test_qwenvl_video_logit_score.py --test_random_image --prompt "a dog" \
        --vllm_url http://127.0.0.1:8000/v1 \
        --model_name /apdcephfs_nj7/share_1220751/xianyihe/ckpts/Qwen/Qwen3-VL-8B-Instruct
"""

import argparse
import sys
import time
import numpy as np
import torch
from PIL import Image


def build_qwenvl_video_logit_score(device, vllm_url, model_name):
    """
    从 rewards.py 中提取的 qwenvl_video_logit_score 函数完整实现，
    这里直接内联以避免导入依赖。
    """
    import asyncio
    import aiohttp
    import base64
    import math
    from io import BytesIO

    SCORE_PROMPT_TEMPLATE = (
        "You are a video/image quality evaluator. Given a video/image and the following text description, "
        "evaluate how well the video/image matches the description and its overall quality.\n\n"
        "Text description: {prompt}\n\n"
        "Please rate the video/image on a scale of 1 to 5, where:\n"
        "1 = Very poor quality or completely irrelevant to the description\n"
        "2 = Poor quality or mostly irrelevant\n"
        "3 = Average quality and somewhat relevant\n"
        "4 = Good quality and mostly matches the description\n"
        "5 = Excellent quality and perfectly matches the description\n\n"
        "Respond with ONLY a single number (1, 2, 3, 4, or 5)."
    )

    SCORE_TOKENS = ["1", "2", "3", "4", "5"]

    def encode_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def compute_weighted_score_from_logprobs(logprobs_data):
        if not logprobs_data or len(logprobs_data) == 0:
            return 0.0

        first_token_logprobs = logprobs_data[0]

        score_logprobs = []
        for token_str in SCORE_TOKENS:
            if token_str in first_token_logprobs:
                score_logprobs.append(first_token_logprobs[token_str])
            else:
                score_logprobs.append(-100.0)

        max_logprob = max(score_logprobs)
        exp_logprobs = [math.exp(lp - max_logprob) for lp in score_logprobs]
        sum_exp = sum(exp_logprobs)
        probabilities = [e / sum_exp for e in exp_logprobs]

        weighted_score = sum((i + 1) * p for i, p in enumerate(probabilities))
        return weighted_score / 5.0

    async def query_vllm_logprobs(session, url, model, prompt_text, image_content):
        messages = [
            {
                "role": "user",
                "content": image_content + [
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 20,
        }

        try:
            async with session.post(
                f"{url}/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"[ERROR] vLLM request failed with status {response.status}: {error_text}")
                    return 0.0

                result = await response.json()
                choices = result.get("choices", [])
                if not choices:
                    return 0.0

                choice = choices[0]
                logprobs_content = choice.get("logprobs", {}).get("content", [])
                if not logprobs_content:
                    text = choice.get("message", {}).get("content", "").strip()
                    if text in SCORE_TOKENS:
                        return float(text) / 5.0
                    return 0.0

                first_token_info = logprobs_content[0]
                top_logprobs_list = first_token_info.get("top_logprobs", [])

                logprobs_dict = {}
                for item in top_logprobs_list:
                    token_str = item.get("token", "")
                    logprob_val = item.get("logprob", -100.0)
                    logprobs_dict[token_str] = logprob_val

                return compute_weighted_score_from_logprobs([logprobs_dict])

        except Exception as e:
            print(f"[ERROR] Error querying vLLM: {e}")
            return 0.0

    async def evaluate_batch(images_data, prompts_text, url, model):
        async with aiohttp.ClientSession() as session:
            tasks = []
            for img_data, prompt in zip(images_data, prompts_text):
                prompt_text = SCORE_PROMPT_TEMPLATE.format(prompt=prompt)
                tasks.append(query_vllm_logprobs(session, url, model, prompt_text, img_data))
            results = await asyncio.gather(*tasks)
            return list(results)

    def _fn(images, prompts, metadata):
        # Convert tensor/numpy to PIL Images
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                images_np = (images.permute(0, 2, 3, 1) * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            elif images.dim() == 5 and images.shape[2] == 3:
                images_np = (images.permute(0, 1, 3, 4, 2) * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            elif images.dim() == 5 and images.shape[1] == 3:
                images_np = (images.permute(0, 2, 3, 4, 1) * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            else:
                images_np = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        elif isinstance(images, np.ndarray):
            if images.ndim == 4 and images.shape[1] == 3:
                images_np = images.transpose(0, 2, 3, 1)
            elif images.ndim == 5 and images.shape[2] == 3:
                images_np = images.transpose(0, 1, 3, 4, 2)
            elif images.ndim == 5 and images.shape[1] == 3:
                images_np = images.transpose(0, 2, 3, 4, 1)
            else:
                images_np = images
            if images_np.dtype != np.uint8:
                images_np = np.clip(images_np * 255, 0, 255).astype(np.uint8)
        else:
            images_np = images

        # Build image content for each sample
        all_image_content = []
        for sample in images_np:
            if isinstance(sample, np.ndarray):
                if sample.ndim == 3:
                    pil_img = Image.fromarray(sample)
                    base64_str = encode_image_to_base64(pil_img)
                    all_image_content.append([
                        {"type": "image_url", "image_url": {"url": base64_str}}
                    ])
                elif sample.ndim == 4:
                    num_frames = sample.shape[0]
                    max_frames = min(8, num_frames)
                    indices = np.linspace(0, num_frames - 1, max_frames, dtype=int)
                    content_items = []
                    for idx in indices:
                        frame = Image.fromarray(sample[idx])
                        base64_str = encode_image_to_base64(frame)
                        content_items.append(
                            {"type": "image_url", "image_url": {"url": base64_str}}
                        )
                    all_image_content.append(content_items)
                else:
                    all_image_content.append([])
            elif isinstance(sample, Image.Image):
                base64_str = encode_image_to_base64(sample)
                all_image_content.append([
                    {"type": "image_url", "image_url": {"url": base64_str}}
                ])
            else:
                all_image_content.append([])

        # Run async evaluation
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    scores = pool.submit(
                        asyncio.run,
                        evaluate_batch(all_image_content, prompts, vllm_url, model_name)
                    ).result()
            else:
                scores = loop.run_until_complete(
                    evaluate_batch(all_image_content, prompts, vllm_url, model_name)
                )
        except RuntimeError:
            scores = asyncio.run(
                evaluate_batch(all_image_content, prompts, vllm_url, model_name)
            )

        return scores, {}

    return _fn


def load_images_as_tensor(image_paths):
    """加载图片列表并转成 (B, C, H, W) tensor, 值域 [0, 1]"""
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    tensors = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        tensors.append(transform(img))
    return torch.stack(tensors)


def load_video_frames_as_tensor(frame_paths, num_frames=None):
    """加载视频帧列表并转成 (1, F, C, H, W) tensor, 值域 [0, 1]"""
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    frames = []
    for path in frame_paths:
        img = Image.open(path).convert("RGB")
        frames.append(transform(img))
    video = torch.stack(frames)  # (F, C, H, W)
    return video.unsqueeze(0)    # (1, F, C, H, W)


def generate_random_image_tensor(batch_size=1, height=512, width=512):
    """生成随机图片 tensor (B, C, H, W), 值域 [0, 1]"""
    return torch.rand(batch_size, 3, height, width)


def generate_random_video_tensor(batch_size=1, num_frames=8, height=512, width=512):
    """生成随机视频 tensor (B, F, C, H, W), 值域 [0, 1]"""
    return torch.rand(batch_size, num_frames, 3, height, width)


def main():
    parser = argparse.ArgumentParser(description="测试 qwenvl_video_logit_score 函数")
    
    # 输入模式（互斥）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", nargs="+", help="输入图片路径（一张或多张）")
    input_group.add_argument("--video_frames", nargs="+", help="输入视频帧路径（多张图片组成视频）")
    input_group.add_argument("--test_random_image", action="store_true", help="使用随机 tensor 测试图片模式")
    input_group.add_argument("--test_random_video", action="store_true", help="使用随机 tensor 测试视频模式")

    # prompt 参数
    parser.add_argument("--prompt", nargs="+", default=["a beautiful scene"],
                        help="文本 prompt（多个 prompt 对应多张图片，单个 prompt 会广播给所有图片/视频）")

    # vLLM 参数
    parser.add_argument("--vllm_url", type=str, default="http://127.0.0.1:8000/v1",
                        help="vLLM 服务的 URL (default: http://127.0.0.1:8000/v1)")
    parser.add_argument("--model_name", type=str,
                        default="/home/ma-user/work/xianyi/ckpts/Qwen/Qwen3-VL-8B-Instruct",
                        help="vLLM 中注册的模型名称")

    # 随机模式参数
    parser.add_argument("--batch_size", type=int, default=1, help="随机模式的 batch size")
    parser.add_argument("--num_frames", type=int, default=8, help="随机视频模式的帧数")
    parser.add_argument("--height", type=int, default=512, help="图片/视频高度")
    parser.add_argument("--width", type=int, default=512, help="图片/视频宽度")

    args = parser.parse_args()

    # =========== 构建输入 ===========
    print("=" * 60)
    print("qwenvl_video_logit_score 测试脚本")
    print("=" * 60)
    print(f"vLLM URL:    {args.vllm_url}")
    print(f"Model Name:  {args.model_name}")
    print()

    if args.image:
        print(f"[模式] 图片输入，共 {len(args.image)} 张")
        images = load_images_as_tensor(args.image)
        prompts = args.prompt if len(args.prompt) == len(args.image) else args.prompt * len(args.image)
        print(f"  图片 tensor shape: {images.shape}  (B, C, H, W)")

    elif args.video_frames:
        print(f"[模式] 视频输入，共 {len(args.video_frames)} 帧")
        images = load_video_frames_as_tensor(args.video_frames)
        prompts = args.prompt[:1]  # 视频只有一个 sample
        print(f"  视频 tensor shape: {images.shape}  (B, F, C, H, W)")

    elif args.test_random_image:
        print(f"[模式] 随机图片 tensor，batch_size={args.batch_size}, size=({args.height}, {args.width})")
        images = generate_random_image_tensor(args.batch_size, args.height, args.width)
        prompts = args.prompt if len(args.prompt) == args.batch_size else args.prompt * args.batch_size
        print(f"  图片 tensor shape: {images.shape}  (B, C, H, W)")

    elif args.test_random_video:
        print(f"[模式] 随机视频 tensor，batch_size={args.batch_size}, num_frames={args.num_frames}, size=({args.height}, {args.width})")
        images = generate_random_video_tensor(args.batch_size, args.num_frames, args.height, args.width)
        prompts = args.prompt if len(args.prompt) == args.batch_size else args.prompt * args.batch_size
        print(f"  视频 tensor shape: {images.shape}  (B, F, C, H, W)")

    print(f"  Prompts: {prompts}")
    print()

    # =========== 构建 scorer 并评分 ===========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] 正在初始化 qwenvl_video_logit_score ...")

    score_fn = build_qwenvl_video_logit_score(
        device=device,
        vllm_url=args.vllm_url,
        model_name=args.model_name,
    )

    print(f"[INFO] 正在调用评分函数 ...")
    start_time = time.time()

    scores, meta = score_fn(images, prompts, {})

    elapsed = time.time() - start_time

    # =========== 打印结果 ===========
    print()
    print("=" * 60)
    print("评分结果")
    print("=" * 60)
    for i, (score, prompt) in enumerate(zip(scores, prompts)):
        raw_score = score * 5.0  # 还原到 1-5 分
        print(f"  样本 {i}: score={score:.4f} (原始分 {raw_score:.2f}/5)  prompt=\"{prompt}\"")
    print()
    print(f"  平均 score: {np.mean(scores):.4f} (原始分 {np.mean(scores)*5:.2f}/5)")
    print(f"  耗时: {elapsed:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
