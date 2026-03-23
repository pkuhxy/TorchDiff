"""
VBench Aesthetic Quality Reward
Based on CLIP ViT-L/14 + LAION Aesthetic Predictor (linear layer).
Score = aesthetic_model(CLIP_features) / 10, normalized to [0, 1].

Input: list of images (PIL or numpy, multi-frame as video) + prompts
Output: list of float rewards
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Union
from PIL import Image

import clip


class VBenchAestheticQualityScorer:
    def __init__(self, device: str = "cuda", cache_dir: str = None):
        """
        Aesthetic quality reward based on VBench implementation.
        Uses CLIP ViT-L/14 + LAION aesthetic linear predictor.
        
        :param device: Device to run the model on
        :param cache_dir: Cache directory for model weights
        """
        self.device = device
        
        if cache_dir is None:
            cache_dir = os.environ.get("VBENCH_CACHE_DIR", os.path.expanduser("~/.cache/vbench"))
        
        # Load CLIP ViT-L/14
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=device)
        self.clip_model.eval()
        
        # Load LAION aesthetic predictor (linear layer: 768 -> 1)
        self.aesthetic_model = self._load_aesthetic_model(cache_dir)
        self.aesthetic_model = self.aesthetic_model.to(device)
        self.aesthetic_model.eval()

        # Build transform: Resize + CenterCrop + Normalize (CLIP standard)
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)
            ),
        ])

    def _load_aesthetic_model(self, cache_folder: str) -> nn.Module:
        """Load the LAION aesthetic predictor linear layer."""
        path_to_model = os.path.join(cache_folder, "sa_0_4_vit_l_14_linear.pth")
        if not os.path.exists(path_to_model):
            os.makedirs(cache_folder, exist_ok=True)
            url_model = (
                "https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/"
                "main/sa_0_4_vit_l_14_linear.pth"
            )
            try:
                from urllib.request import urlretrieve
                print(f"Downloading aesthetic model to {path_to_model}...")
                urlretrieve(url_model, path_to_model)
            except Exception as e:
                import subprocess
                print(f"urlretrieve failed: {e}, trying wget...")
                subprocess.run(["wget", "-O", path_to_model, url_model], check=True)
        
        model = nn.Linear(768, 1)
        state_dict = torch.load(path_to_model, map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    def _images_to_tensors(self, images: Union[List[Image.Image], List[np.ndarray], np.ndarray]) -> List[torch.Tensor]:
        """
        Convert input images to a list of tensors.
        Each element: [N_frames, C, H, W] float tensor in [0, 1].
        
        Supports:
        - List[PIL.Image] -> each is a single frame
        - List[np.ndarray] -> each can be (H,W,C) single frame or (F,H,W,C) multi-frame video
        - np.ndarray -> shape (B,H,W,C) or (B,F,H,W,C)
        """
        result = []
        if isinstance(images, np.ndarray):
            if images.ndim == 4:  # (B, H, W, C) batch of single frames
                for i in range(images.shape[0]):
                    t = torch.from_numpy(images[i]).permute(2, 0, 1).float() / 255.0  # (C, H, W)
                    result.append(t.unsqueeze(0))  # (1, C, H, W)
            elif images.ndim == 5:  # (B, F, H, W, C) batch of videos
                for i in range(images.shape[0]):
                    t = torch.from_numpy(images[i]).permute(0, 3, 1, 2).float() / 255.0  # (F, C, H, W)
                    result.append(t)
            else:
                raise ValueError(f"Unexpected numpy array shape: {images.shape}")
        else:
            for img in images:
                if isinstance(img, Image.Image):
                    arr = np.array(img)
                    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                    result.append(t.unsqueeze(0))
                elif isinstance(img, np.ndarray):
                    if img.ndim == 3:  # (H, W, C)
                        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                        result.append(t.unsqueeze(0))
                    elif img.ndim == 4:  # (F, H, W, C)
                        t = torch.from_numpy(img).permute(0, 3, 1, 2).float() / 255.0
                        result.append(t)
                    else:
                        raise ValueError(f"Unexpected image ndim: {img.ndim}")
                elif isinstance(img, torch.Tensor):
                    if img.ndim == 3:  # (C, H, W)
                        result.append(img.unsqueeze(0))
                    elif img.ndim == 4:  # (F, C, H, W)
                        result.append(img)
                    else:
                        raise ValueError(f"Unexpected tensor ndim: {img.ndim}")
        return result

    @torch.no_grad()
    def __call__(
        self,
        images: Union[List[Image.Image], List[np.ndarray], np.ndarray],
        prompts: List[str],
    ) -> List[float]:
        """
        Calculate aesthetic quality reward.
        
        :param images: List of images (PIL, numpy HWC, or numpy FHWC for video frames)
        :param prompts: List of text prompts (not directly used, kept for interface consistency)
        :return: List of reward scores in [0, 1]
        """
        assert len(images) == len(prompts), "Images and prompts must have the same length"
        
        frame_tensors = self._images_to_tensors(images)
        rewards = []
        batch_size = 32
        
        for frames in frame_tensors:
            # frames: (N, C, H, W) in [0, 1]
            frames_transformed = self.transform(frames).to(self.device)
            
            all_scores = []
            for i in range(0, len(frames_transformed), batch_size):
                batch = frames_transformed[i:i + batch_size]
                image_feats = self.clip_model.encode_image(batch).to(torch.float32)
                image_feats = F.normalize(image_feats, dim=-1, p=2)
                scores = self.aesthetic_model(image_feats).squeeze(dim=-1)
                all_scores.append(scores)
            
            all_scores = torch.cat(all_scores, dim=0)
            # Normalize: VBench divides by 10
            normalized = all_scores / 10.0
            avg_score = normalized.mean().item()
            # Clamp to [0, 1]
            avg_score = max(0.0, min(1.0, avg_score))
            rewards.append(avg_score)
        
        return rewards


if __name__ == "__main__":
    scorer = VBenchAestheticQualityScorer(device="cpu")
    test_image = Image.new("RGB", (512, 512), (128, 128, 200))
    reward = scorer([test_image], ["a beautiful landscape"])
    print(f"Aesthetic Quality Reward: {reward}")
