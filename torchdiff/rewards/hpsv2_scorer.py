"""
Human Preference Score (HPS) Reward
Based on HPSv2 library (supports v2.0 and v2.1).
HPSv3 code is not yet publicly available; this implementation uses HPSv2.1
which is currently the best available HPS model.

When HPSv3 code is released, this file can be updated to support it.

NOTE: 不使用 hpsv2.score() 高层 API，因为它内部硬编码 device='cuda'（即 cuda:0），
      在多卡分布式训练中，所有 rank 的 reward model 都会加载到 npu0 上导致 OOM。
      本实现手动加载模型到指定 device 上，避免此问题。

Input: list of images (PIL or numpy) + prompts
Output: list of float rewards
"""

import os
import numpy as np
import torch
from typing import List, Union
from PIL import Image


def _load_hps_model(device, hps_version="v2.1"):
    """
    手动加载 HPSv2 模型到指定 device 上，避免 hpsv2.score() 默认加载到 cuda:0 的问题。
    
    返回: (model, preprocess_val, tokenizer)
    """
    try:
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    except ImportError:
        raise ImportError(
            "hpsv2 package is required. Install with: pip install hpsv2\n"
            "See: https://github.com/tgxs002/HPSv2"
        )
    
    # 创建模型架构并加载到指定 device
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        'ViT-H-14',
        'laion2B-s32B-b79K',
        precision='amp',
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False,
    )
    
    # 加载微调权重
    # 确定模型权重路径
    root_path = os.environ.get("HPS_ROOT", os.path.expanduser("~/.cache/hpsv2"))
    if hps_version == "v2.0":
        cp_name = "HPS_v2_compressed.pt"
    else:
        cp_name = "HPS_v2.1_compressed.pt"
    
    cp_path = os.path.join(root_path, cp_name)
    
    if not os.path.exists(cp_path):
        # 尝试通过 huggingface_hub 下载
        try:
            from huggingface_hub import hf_hub_download
            cp_path = hf_hub_download(repo_id="xswu/HPSv2", filename=cp_name, cache_dir=root_path)
        except Exception:
            raise FileNotFoundError(
                f"HPS model weights not found at {cp_path}. "
                f"Please download from https://huggingface.co/xswu/HPSv2 "
                f"or set HPS_ROOT environment variable."
            )
    
    # 加载权重到指定 device（关键：map_location 到正确 device）
    checkpoint = torch.load(cp_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
    tokenizer = get_tokenizer('ViT-H-14')
    
    return model, preprocess_val, tokenizer


class HPSScorer:
    def __init__(self, device: str = "cuda", hps_version: str = "v2.1"):
        """
        Human Preference Score reward.
        手动加载模型到指定 device 上，避免 hpsv2.score() 默认加载到 cuda:0 的问题。
        
        :param device: Device to run the model on (e.g. 'cuda:0', 'cuda:3')
        :param hps_version: HPS version to use ('v2.0', 'v2.1')
        """
        self.device = device
        self.hps_version = hps_version
        
        self.model, self.preprocess_val, self.tokenizer = _load_hps_model(device, hps_version)

    def _ensure_pil_images(
        self, images: Union[List[Image.Image], List[np.ndarray], np.ndarray]
    ) -> List[Image.Image]:
        """Convert various image formats to PIL Images."""
        pil_images = []
        if isinstance(images, np.ndarray):
            if images.ndim == 4:  # (B, H, W, C) or (B, F, H, W, C)
                for i in range(images.shape[0]):
                    pil_images.append(Image.fromarray(images[i].astype(np.uint8)))
            elif images.ndim == 5:  # (B, F, H, W, C) - take middle frame
                for i in range(images.shape[0]):
                    mid_idx = images.shape[1] // 2
                    pil_images.append(Image.fromarray(images[i, mid_idx].astype(np.uint8)))
        else:
            for img in images:
                if isinstance(img, Image.Image):
                    pil_images.append(img)
                elif isinstance(img, np.ndarray):
                    if img.ndim == 3:  # (H, W, C)
                        pil_images.append(Image.fromarray(img.astype(np.uint8)))
                    elif img.ndim == 4:  # (F, H, W, C) - take middle frame
                        mid_idx = img.shape[0] // 2
                        pil_images.append(Image.fromarray(img[mid_idx].astype(np.uint8)))
                elif isinstance(img, torch.Tensor):
                    if img.ndim == 3:  # (C, H, W)
                        arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        pil_images.append(Image.fromarray(arr))
                    elif img.ndim == 4:  # (F, C, H, W) - take middle frame
                        mid_idx = img.shape[0] // 2
                        arr = (img[mid_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        pil_images.append(Image.fromarray(arr))
        return pil_images

    @torch.no_grad()
    def __call__(
        self,
        images: Union[List[Image.Image], List[np.ndarray], np.ndarray],
        prompts: List[str],
    ) -> List[float]:
        """
        Calculate HPS reward.
        
        :param images: List of images (PIL, numpy HWC, or numpy FHWC for video)
                       For video inputs, the middle frame is used for scoring.
        :param prompts: List of text prompts
        :return: List of reward scores
        """
        assert len(images) == len(prompts), "Images and prompts must have the same length"
        
        pil_images = self._ensure_pil_images(images)
        rewards = []
        
        for img, prompt in zip(pil_images, prompts):
            # 手动做预处理 + 推理，使用 self.device
            image_tensor = self.preprocess_val(img).unsqueeze(0).to(device=self.device, non_blocking=True)
            text_tensor = self.tokenizer([prompt]).to(device=self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                outputs = self.model(image_tensor, text_tensor)
                image_features = outputs["image_features"]
                text_features = outputs["text_features"]
                logits_per_image = image_features @ text_features.T
                hps_score = torch.diagonal(logits_per_image).cpu().numpy()
            
            rewards.append(float(hps_score[0]))
        
        return rewards


class HPSScorer_video_or_image:
    def __init__(self, device: str = "cuda", hps_version: str = "v2.1"):
        """
        HPS reward calculator for both images and videos.
        For videos, scores multiple sampled frames and averages.
        手动加载模型到指定 device 上，避免 hpsv2.score() 默认加载到 cuda:0 的问题。
        
        :param device: Device to run the model on (e.g. 'cuda:0', 'cuda:3')
        :param hps_version: HPS version ('v2.0', 'v2.1')
        """
        self.device = device
        self.hps_version = hps_version
        self.frame_interval = 4  # Sample every 4th frame for videos
        
        self.model, self.preprocess_val, self.tokenizer = _load_hps_model(device, hps_version)

    def _score_single_image(self, pil_img: Image.Image, prompt: str) -> float:
        """对单张 PIL 图像计算 HPS 分数。"""
        image_tensor = self.preprocess_val(pil_img).unsqueeze(0).to(device=self.device, non_blocking=True)
        text_tensor = self.tokenizer([prompt]).to(device=self.device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            outputs = self.model(image_tensor, text_tensor)
            image_features = outputs["image_features"]
            text_features = outputs["text_features"]
            logits_per_image = image_features @ text_features.T
            hps_score = torch.diagonal(logits_per_image).cpu().numpy()
        
        return float(hps_score[0])

    @torch.no_grad()
    def __call__(
        self,
        images: Union[List[Image.Image], List[np.ndarray]],
        prompts: List[str],
    ) -> List[float]:
        """
        Calculate HPS reward for images or videos.
        
        :param images: List of images or videos
                       - PIL.Image: single image
                       - np.ndarray (H,W,C): single image
                       - np.ndarray (F,H,W,C): video
        :param prompts: List of text prompts
        :return: List of reward scores
        """
        assert len(images) == len(prompts), "Images and prompts must have the same length"
        
        rewards = []
        for img, prompt in zip(images, prompts):
            frame_scores = []
            
            # Handle video: shape (F, H, W, C)
            if isinstance(img, np.ndarray) and img.ndim == 4:
                sampled_frames = img[::self.frame_interval]
                for frame in sampled_frames:
                    pil_frame = Image.fromarray(frame.astype(np.uint8))
                    score = self._score_single_image(pil_frame, prompt)
                    frame_scores.append(score)
            else:
                # Single image
                if isinstance(img, Image.Image):
                    pil_img = img
                elif isinstance(img, np.ndarray):
                    pil_img = Image.fromarray(img.astype(np.uint8))
                elif isinstance(img, torch.Tensor):
                    arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    pil_img = Image.fromarray(arr)
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")
                
                score = self._score_single_image(pil_img, prompt)
                frame_scores.append(score)
            
            if frame_scores:
                rewards.append(sum(frame_scores) / len(frame_scores))
            else:
                rewards.append(0.0)
        
        return rewards


if __name__ == "__main__":
    scorer = HPSScorer(device="cuda", hps_version="v2.1")
    test_image = Image.new("RGB", (512, 512), (128, 128, 200))
    reward = scorer([test_image], ["a beautiful mountain landscape at sunset"])
    print(f"HPS Reward: {reward}")
