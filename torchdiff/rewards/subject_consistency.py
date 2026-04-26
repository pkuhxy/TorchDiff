"""
VBench Subject Consistency Reward
Based on DINO ViT-B/16 (facebookresearch/dino).
Computes cosine similarity between consecutive frames and the first frame,
averaged as (sim_prev + sim_first) / 2 per frame.

Input: list of multi-frame images (as video frames) + prompts
Output: list of float rewards in [0, 1]
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from typing import List, Union
from PIL import Image


class VBenchSubjectConsistencyScorer:
    def __init__(self, device: str = "cuda", dino_model_path: str = None):
        """
        Subject consistency reward based on VBench implementation.
        Uses DINOv2 ViT-B to compute frame-to-frame feature similarity.
        
        :param device: Device to run the model on
        :param dino_model_path: Path to local HuggingFace DINOv2 model directory.
                                If None, downloads 'facebook/dinov2-base' from HuggingFace Hub.
        """
        self.device = device
        
        # Load DINOv2
        from transformers import AutoModel
        model_name_or_path = dino_model_path if dino_model_path is not None else "facebook/dinov2-base"
        self.model = AutoModel.from_pretrained(model_name_or_path).to(device)
        self.model.eval()
        
        # DINO transform (ImageNet normalization, no center crop)
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)
            ),
        ])

    def _images_to_frame_tensors(
        self, images: Union[List[Image.Image], List[np.ndarray], np.ndarray]
    ) -> List[torch.Tensor]:
        """
        Convert images to list of frame tensors.
        Each element: (N, C, H, W) float tensor in [0, 1].
        """
        result = []
        if isinstance(images, np.ndarray):
            if images.ndim == 4:  # (B, H, W, C)
                for i in range(images.shape[0]):
                    t = torch.from_numpy(images[i]).permute(2, 0, 1).float() / 255.0
                    result.append(t.unsqueeze(0))
            elif images.ndim == 5:  # (B, F, H, W, C)
                for i in range(images.shape[0]):
                    t = torch.from_numpy(images[i]).permute(0, 3, 1, 2).float() / 255.0
                    result.append(t)
        else:
            for img in images:
                if isinstance(img, Image.Image):
                    arr = np.array(img)
                    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                    result.append(t.unsqueeze(0))
                elif isinstance(img, np.ndarray):
                    if img.ndim == 3:
                        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                        result.append(t.unsqueeze(0))
                    elif img.ndim == 4:
                        t = torch.from_numpy(img).permute(0, 3, 1, 2).float() / 255.0
                        result.append(t)
                elif isinstance(img, torch.Tensor):
                    if img.ndim == 3:
                        result.append(img.unsqueeze(0))
                    elif img.ndim == 4:
                        result.append(img)
        return result

    def _compute_consistency(self, frames: torch.Tensor) -> float:
        """
        Compute subject consistency score across frames.
        
        :param frames: (N, C, H, W) float tensor in [0, 1]
        :return: Consistency score in [0, 1]
        """
        if len(frames) < 2:
            return 1.0  # Single frame => perfectly consistent
        
        # Apply DINO transform
        frames_transformed = self.transform(frames)
        
        total_sim = 0.0
        first_features = None
        former_features = None
        
        for i in range(len(frames_transformed)):
            image = frames_transformed[i:i+1].to(self.device)
            output = self.model(image)
            # DINOv2 via transformers: use CLS token (last_hidden_state[:, 0])
            features = output.last_hidden_state[:, 0]
            features = F.normalize(features, dim=-1, p=2)
            
            if i == 0:
                first_features = features
            else:
                sim_prev = max(0.0, F.cosine_similarity(former_features, features).item())
                sim_first = max(0.0, F.cosine_similarity(first_features, features).item())
                cur_sim = (sim_prev + sim_first) / 2.0
                total_sim += cur_sim
            
            former_features = features
        
        avg_sim = total_sim / (len(frames_transformed) - 1)
        return avg_sim

    @torch.no_grad()
    def __call__(
        self,
        images: Union[List[Image.Image], List[np.ndarray], np.ndarray],
        prompts: List[str],
    ) -> List[float]:
        """
        Calculate subject consistency reward.
        
        :param images: List of images. For video: each element is np.ndarray (F,H,W,C).
                       For single image: returns 1.0 (trivially consistent).
        :param prompts: List of text prompts (not used, kept for interface consistency)
        :return: List of reward scores in [0, 1]
        """
        assert len(images) == len(prompts), "Images and prompts must have the same length"
        
        frame_tensors = self._images_to_frame_tensors(images)
        rewards = []
        
        for frames in frame_tensors:
            score = self._compute_consistency(frames)
            rewards.append(score)
        
        return rewards


if __name__ == "__main__":
    scorer = VBenchSubjectConsistencyScorer(device="cpu", dino_model_path="/home/ma-user/work/xianyi/ckpts/facebook/dinov2-base")
    # Simulate a 4-frame video
    frames = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(4)]
    video = np.stack(frames)  # (4, 256, 256, 3)
    reward = scorer([video], ["a dog sitting on a table"])
    print(f"Subject Consistency Reward: {reward}")
