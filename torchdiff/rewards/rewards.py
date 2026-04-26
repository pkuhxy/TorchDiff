from PIL import Image
import io
import numpy as np
import torch
from collections import defaultdict

def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew/500, meta

    return _fn

def aesthetic_score():
    from torchdiff.rewards.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn

def clip_score(device):
    from torchdiff.rewards.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def image_similarity_score(device):
    from torchdiff.rewards.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device).cuda()

    def _fn(images, ref_images):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        if not isinstance(ref_images, torch.Tensor):
            ref_images = [np.array(img) for img in ref_images]
            ref_images = np.array(ref_images)
            ref_images = ref_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            ref_images = torch.tensor(ref_images, dtype=torch.uint8)/255.0
        scores = scorer.image_similarity(images, ref_images)
        return scores, {}

    return _fn

def pickscore_score(device):
    from torchdiff.rewards.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def imagereward_score(device):
    from torchdiff.rewards.imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def qwenvl_score(device):
    from torchdiff.rewards.qwenvl import QwenVLScorer

    scorer = QwenVLScorer(dtype=torch.bfloat16, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

    
def ocr_score(device):
    from torchdiff.rewards.ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def video_ocr_score(device):
    import sys
    print(f"[video_ocr_score] Importing OcrScorer_video_or_image ...", flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    from torchdiff.rewards.ocr import OcrScorer_video_or_image

    print(f"[video_ocr_score] Creating OcrScorer_video_or_image instance ...", flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    scorer = OcrScorer_video_or_image()
    print(f"[video_ocr_score] OcrScorer_video_or_image initialized successfully.", flush=True)
    sys.stdout.flush(); sys.stderr.flush()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                images = images.permute(0, 2, 3, 1) 
            elif images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        elif isinstance(images, np.ndarray):
            # Handle numpy arrays with channel-first layout (e.g. from .cpu().numpy())
            if images.ndim == 4 and images.shape[1] == 3:
                # (B, C, H, W) -> (B, H, W, C)
                images = images.transpose(0, 2, 3, 1)
            elif images.ndim == 5 and images.shape[1] == 3:
                # (B, C, T, H, W) -> (B, T, H, W, C)
                images = images.transpose(0, 2, 3, 4, 1)
            elif images.ndim == 5 and images.shape[2] == 3:
                # (B, T, C, H, W) -> (B, T, H, W, C)
                images = images.transpose(0, 1, 3, 4, 2)
            # Convert float [0, 1] to uint8 [0, 255] if needed
            if images.dtype != np.uint8:
                images = np.clip(images * 255, 0, 255).astype(np.uint8)
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def deqa_score_remote(device):
    """Submits images to DeQA and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://127.0.0.1:18086"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        all_scores = []
        for image_batch in images_batched:
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn

def geneval_score(device):
    """Submits images to GenEval and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://127.0.0.1:18085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadatas, only_strict):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "meta_datas": list(metadata_batched),
                "only_strict": only_strict,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            all_scores += response_data["scores"]
            all_rewards += response_data["rewards"]
            all_strict_rewards += response_data["strict_rewards"]
            all_group_strict_rewards.append(response_data["group_strict_rewards"])
            all_group_rewards.append(response_data["group_rewards"])
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)

        return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict

    return _fn

def unifiedreward_score_remote(device):
    """Submits images to DeQA and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://10.82.120.15:18085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "prompts": prompt_batch
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            print("response: ", response)
            print("response: ", response.content)
            response_data = pickle.loads(response.content)

            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn

def unifiedreward_score_sglang(device):
    import asyncio
    from openai import AsyncOpenAI
    import base64
    from io import BytesIO
    import re 

    def pil_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    def _extract_scores(text_outputs):
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    client = AsyncOpenAI(base_url="http://127.0.0.1:17140/v1", api_key="flowgrpo")
        
    async def evaluate_image(prompt, image):
        question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after \'Final Score:\'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
        images_base64 = pil_image_to_base64(image)
        response = await client.chat.completions.create(
            model="UnifiedReward-7b-v1.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": images_base64},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    async def evaluate_batch_image(images, prompts):
        tasks = [evaluate_image(prompt, img) for prompt, img in zip(prompts, images)]
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts, metadata):
        # 处理Tensor类型转换
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        # 转换为PIL Image并调整尺寸
        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        # 执行异步批量评估
        text_outputs = asyncio.run(evaluate_batch_image(images, prompts))
        score = _extract_scores(text_outputs)
        score = [sc/5.0 for sc in score]
        return score, {}
    
    return _fn

def qwenvl_video_logit_score(device, vllm_url="http://127.0.0.1:8000/v1", model_name="/home/ma-user/work/xianyi/ckpts/Qwen/Qwen3-VL-32B-Instruct"):
    """
    QwenVL Video/Image reward using vLLM server.
    
    Sends video/image + prompt to a vLLM-served Qwen-VL model, asks it to score 1-5,
    and computes a soft reward from the logits of tokens "1","2","3","4","5"
    via weighted sum: reward = sum(i * softmax(logit_i)) for i in {1,2,3,4,5}.
    
    The vLLM server should be started separately, e.g.:
        python -m vllm.entrypoints.openai.api_server \
            --model Qwen/Qwen2.5-VL-7B-Instruct \
            --tensor-parallel-size 1 \
            --port 8000 \
            --trust-remote-code
    
    Args:
        device: not used directly (vLLM runs on its own GPUs), kept for interface consistency.
        vllm_url: URL of the vLLM OpenAI-compatible server.
        model_name: The model name as registered in vLLM.
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

    # Token IDs for "1","2","3","4","5" in Qwen2.5 tokenizer
    # We'll query logprobs for all tokens and pick the ones we need
    SCORE_TOKENS = ["1", "2", "3", "4", "5"]

    def encode_image_to_base64(image):
        """Encode a PIL Image to base64 data URI."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def encode_video_frames_to_base64(frames):
        """Encode a list of PIL Images (video frames) to base64 data URIs."""
        encoded_frames = []
        for frame in frames:
            encoded_frames.append(encode_image_to_base64(frame))
        return encoded_frames

    def compute_weighted_score_from_logprobs(logprobs_data):
        """
        Given the logprobs output from vLLM for the first generated token,
        extract logits for tokens "1"-"5", apply softmax, and compute weighted sum.
        
        logprobs_data: list of dicts, each containing top logprobs for a token position.
        We only care about the first token position.
        
        Returns: float score in [1, 5], normalized to [0, 1] by dividing by 5.
        """
        if not logprobs_data or len(logprobs_data) == 0:
            return 0.0

        first_token_logprobs = logprobs_data[0]
        # first_token_logprobs is a dict: {token_str: logprob_value, ...}

        # Extract logprobs for score tokens "1"-"5"
        score_logprobs = []
        for token_str in SCORE_TOKENS:
            if token_str in first_token_logprobs:
                score_logprobs.append(first_token_logprobs[token_str])
            else:
                # If the token is not in top logprobs, assign a very low value
                score_logprobs.append(-100.0)

        # Softmax over the 5 logprobs
        max_logprob = max(score_logprobs)
        exp_logprobs = [math.exp(lp - max_logprob) for lp in score_logprobs]
        sum_exp = sum(exp_logprobs)
        probabilities = [e / sum_exp for e in exp_logprobs]

        # Weighted sum: score = sum(i * p_i) for i in {1,2,3,4,5}
        weighted_score = sum((i + 1) * p for i, p in enumerate(probabilities))

        # Normalize to [0, 1]
        return weighted_score / 5.0

    async def query_vllm_logprobs(session, url, model, prompt_text, image_content):
        """
        Send a single request to vLLM and get logprobs for the first generated token.
        
        Args:
            session: aiohttp session
            url: vLLM API URL (e.g. http://127.0.0.1:8000/v1)
            model: model name
            prompt_text: the scoring prompt
            image_content: list of content items (image_url dicts) for the message
        
        Returns: float score
        """
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
            "top_logprobs": 20,  # Request top-20 logprobs to maximize chance of capturing 1-5
        }

        try:
            async with session.post(
                f"{url}/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"[qwenvl_video_logit_score] vLLM request failed with status {response.status}: {error_text}")
                    return 0.0

                result = await response.json()
                choices = result.get("choices", [])
                if not choices:
                    return 0.0

                choice = choices[0]
                logprobs_content = choice.get("logprobs", {}).get("content", [])
                if not logprobs_content:
                    # Fallback: try to parse the generated text
                    text = choice.get("message", {}).get("content", "").strip()
                    if text in SCORE_TOKENS:
                        return float(text) / 5.0
                    return 0.0

                # Extract the top_logprobs from the first token
                first_token_info = logprobs_content[0]
                top_logprobs_list = first_token_info.get("top_logprobs", [])

                # Convert to dict: {token_str: logprob}
                logprobs_dict = {}
                for item in top_logprobs_list:
                    token_str = item.get("token", "")
                    logprob_val = item.get("logprob", -100.0)
                    logprobs_dict[token_str] = logprob_val

                return compute_weighted_score_from_logprobs([logprobs_dict])

        except Exception as e:
            print(f"[qwenvl_video_logit_score] Error querying vLLM: {e}")
            return 0.0

    async def evaluate_batch(images_data, prompts_text, url, model):
        """Evaluate a batch of images/videos asynchronously."""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for img_data, prompt in zip(images_data, prompts_text):
                prompt_text = SCORE_PROMPT_TEMPLATE.format(prompt=prompt)
                tasks.append(query_vllm_logprobs(session, url, model, prompt_text, img_data))
            results = await asyncio.gather(*tasks)
            return list(results)

    def _fn(images, prompts, metadata):
        """
        Main reward function.
        
        Args:
            images: Tensor (B, C, H, W) for images or (B, F, C, H, W) for videos,
                    or numpy array in NHWC / NFHWC format.
            prompts: List of text prompts.
            metadata: dict (not used).
        
        Returns:
            (scores, {}) where scores is a list of float rewards.
        """
        # Convert tensor/numpy to PIL Images
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                # (B, C, H, W) -> images
                images_np = (images.permute(0, 2, 3, 1) * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            elif images.dim() == 5 and images.shape[2] == 3:
                # (B, F, C, H, W) -> videos
                images_np = (images.permute(0, 1, 3, 4, 2) * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            elif images.dim() == 5 and images.shape[1] == 3:
                # (B, C, F, H, W) -> videos
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
                    # Single image (H, W, C)
                    pil_img = Image.fromarray(sample)
                    base64_str = encode_image_to_base64(pil_img)
                    all_image_content.append([
                        {"type": "image_url", "image_url": {"url": base64_str}}
                    ])
                elif sample.ndim == 4:
                    # Video (F, H, W, C) - sample frames uniformly (max 8 frames)
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
                # If we're already in an async context, create a new loop in a thread
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


def aesthetic_quality_score(device):
    from torchdiff.rewards.aesthetic_quality import VBenchAestheticQualityScorer

    scorer = VBenchAestheticQualityScorer(device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        elif isinstance(images, np.ndarray):
            if images.ndim == 4 and images.shape[1] == 3:
                # (B, C, H, W) -> (B, H, W, C)
                images = images.transpose(0, 2, 3, 1)
            elif images.ndim == 5 and images.shape[1] == 3:
                # (B, C, T, H, W) -> (B, T, H, W, C)
                images = images.transpose(0, 2, 3, 4, 1)
            if images.dtype != np.uint8:
                images = np.clip(images * 255, 0, 255).astype(np.uint8)
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def dynamic_degree_score(device):
    from torchdiff.rewards.dynamic_degree import VBenchDynamicDegreeScorer

    scorer = VBenchDynamicDegreeScorer(device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                images = images.permute(0, 2, 3, 1)
            elif images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        elif isinstance(images, np.ndarray):
            if images.ndim == 4 and images.shape[1] == 3:
                # (B, C, H, W) -> (B, H, W, C)
                images = images.transpose(0, 2, 3, 1)
            elif images.ndim == 5 and images.shape[1] == 3:
                # (B, C, T, H, W) -> (B, T, H, W, C)
                images = images.transpose(0, 2, 3, 4, 1)
            if images.dtype != np.uint8:
                images = np.clip(images * 255, 0, 255).astype(np.uint8)
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def subject_consistency_score(device):
    from torchdiff.rewards.subject_consistency import VBenchSubjectConsistencyScorer

    scorer = VBenchSubjectConsistencyScorer(device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                images = images.permute(0, 2, 3, 1)
            elif images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        elif isinstance(images, np.ndarray):
            if images.ndim == 4 and images.shape[1] == 3:
                # (B, C, H, W) -> (B, H, W, C)
                images = images.transpose(0, 2, 3, 1)
            elif images.ndim == 5 and images.shape[1] == 3:
                # (B, C, T, H, W) -> (B, T, H, W, C)
                images = images.transpose(0, 2, 3, 4, 1)
            if images.dtype != np.uint8:
                images = np.clip(images * 255, 0, 255).astype(np.uint8)
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def overall_consistency_score(device):
    from torchdiff.rewards.overall_consistency import VBenchOverallConsistencyScorer

    scorer = VBenchOverallConsistencyScorer(device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                images = images.permute(0, 2, 3, 1)
            elif images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        elif isinstance(images, np.ndarray):
            if images.ndim == 4 and images.shape[1] == 3:
                # (B, C, H, W) -> (B, H, W, C)
                images = images.transpose(0, 2, 3, 1)
            elif images.ndim == 5 and images.shape[1] == 3:
                # (B, C, T, H, W) -> (B, T, H, W, C)
                images = images.transpose(0, 2, 3, 4, 1)
            if images.dtype != np.uint8:
                images = np.clip(images * 255, 0, 255).astype(np.uint8)
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def videoalign_score(device, load_from_pretrained="/home/ma-user/work/xianyi/ckpts/KlingTeam/VideoReward", reward_dim="Overall", use_norm=True, 
                     videoalign_root=None, save_fps=8, num_frames=None, fps=None, max_pixels=None):
    """
    VideoAlign video quality reward.
    Evaluates videos on VQ (Visual Quality), MQ (Motion Quality), TA (Text Alignment).
    Returns the specified dimension score (default: Overall = VQ + MQ + TA).
    
    Args:
        device: Device string (e.g. "cuda:0")
        load_from_pretrained: Path to VideoAlign checkpoint directory
        reward_dim: Which dimension to return ("VQ", "MQ", "TA", "Overall")
        use_norm: Whether to normalize scores
        videoalign_root: Path to VideoAlign source code root
        save_fps: FPS for writing temp video files
        num_frames: Number of frames for VideoAlign sampling
        fps: FPS for VideoAlign video sampling
        max_pixels: Max pixels for VideoAlign
    """
    from torchdiff.rewards.videoalign_scorer import VideoAlignScorer

    scorer = VideoAlignScorer(
        device=device,
        load_from_pretrained=load_from_pretrained,
        reward_dim=reward_dim,
        use_norm=use_norm,
        save_fps=save_fps,
        num_frames=num_frames,
        fps=fps,
        max_pixels=max_pixels,
    )

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                images = images.permute(0, 2, 3, 1)
            elif images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            elif images.dim() == 5 and images.shape[1] == 3:
                images = images.permute(0, 2, 3, 4, 1)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        elif isinstance(images, np.ndarray):
            if images.ndim == 4 and images.shape[1] == 3:
                images = images.transpose(0, 2, 3, 1)
            elif images.ndim == 5 and images.shape[2] == 3:
                images = images.transpose(0, 1, 3, 4, 2)
            elif images.ndim == 5 and images.shape[1] == 3:
                images = images.transpose(0, 2, 3, 4, 1)
            if images.dtype != np.uint8:
                images = np.clip(images * 255, 0, 255).astype(np.uint8)
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def hps_score(device):
    from torchdiff.rewards.hpsv2_scorer import HPSScorer

    scorer = HPSScorer(device=device, hps_version="v2.1")

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):

            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def video_hps_score(device):
    from torchdiff.rewards.hpsv2_scorer import HPSScorer_video_or_image

    scorer = HPSScorer_video_or_image(device=device, hps_version="v2.1")

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                images = images.permute(0, 2, 3, 1)
            elif images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        elif isinstance(images, np.ndarray):
            # Handle numpy arrays with channel-first layout (e.g. from .cpu().numpy())
            if images.ndim == 4 and images.shape[1] == 3:
                # (B, C, H, W) -> (B, H, W, C)
                images = images.transpose(0, 2, 3, 1)
            elif images.ndim == 5 and images.shape[1] == 3:
                # (B, C, T, H, W) -> (B, T, H, W, C)
                images = images.transpose(0, 2, 3, 4, 1)
            # Convert float [0, 1] to uint8 [0, 255] if needed
            if images.dtype != np.uint8:
                images = np.clip(images * 255, 0, 255).astype(np.uint8)
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def multi_score(device, score_dict):
    score_functions = {
        "deqa": deqa_score_remote,
        "ocr": ocr_score,
        "video_ocr": video_ocr_score,
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "qwenvl": qwenvl_score,
        "aesthetic": aesthetic_score,
        "jpeg_compressibility": jpeg_compressibility,
        "unifiedreward": unifiedreward_score_sglang,
        "geneval": geneval_score,
        "clipscore": clip_score,
        "image_similarity": image_similarity_score,
        "aesthetic_quality": aesthetic_quality_score,
        "dynamic": dynamic_degree_score,
        "subject_consistency": subject_consistency_score,
        "overall_consistency": overall_consistency_score,
        "hps": hps_score,
        "video_hps": video_hps_score,
        "qwenvl_video_logit": qwenvl_video_logit_score,
        "videoalign": videoalign_score,
    }
    import sys
    score_fns={}
    for score_name, weight in score_dict.items():
        print(f"[multi_score] Initializing score function: {score_name} (weight={weight}) ...", flush=True)
        sys.stdout.flush(); sys.stderr.flush()
        try:
            score_fns[score_name] = score_functions[score_name](device) if 'device' in score_functions[score_name].__code__.co_varnames else score_functions[score_name]()
            print(f"[multi_score] Score function {score_name} initialized successfully.", flush=True)
            sys.stdout.flush(); sys.stderr.flush()
        except Exception as e:
            print(f"[multi_score] ERROR initializing {score_name}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.stdout.flush(); sys.stderr.flush()
            raise

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, ref_images=None, only_strict=True):
        total_scores = []
        score_details = {}
        
        for score_name, weight in score_dict.items():
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](images, prompts, metadata, only_strict)
                score_details['accuracy'] = rewards
                score_details['strict_accuracy'] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f'{key}_strict_accuracy'] = value
                for key, value in group_rewards.items():
                    score_details[f'{key}_accuracy'] = value
            elif score_name == "image_similarity":
                scores, rewards = score_fns[score_name](images, ref_images)
            else:
                scores, rewards = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]
            
            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]
        
        score_details['avg'] = total_scores
        return score_details, {}

    return _fn

def main():
    import torchvision.transforms as transforms

    image_paths = [
        "nasa.jpg",
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])

    images = torch.stack([transform(Image.open(image_path).convert('RGB')) for image_path in image_paths])
    prompts=[
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    metadata = {}  # Example metadata
    score_dict = {
        "unifiedreward": 1.0
    }
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores, _ = scoring_fn(images, prompts, metadata)
    # Print the scores
    print("Scores:", scores)


if __name__ == "__main__":
    main()
