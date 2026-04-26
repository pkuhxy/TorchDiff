"""
VBench Overall Consistency Reward
Based on ViCLIP (ViClip-InternVid-10M-FLT) video-language model.
Computes cosine similarity between video features and text features.

Input: list of multi-frame images (as video frames) + prompts
Output: list of float rewards
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "16"  # 限制在 128 以下，通常 8 或 16 性能最好
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["VECLIB_MAXIMUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from typing import List, Union
from PIL import Image


class VBenchOverallConsistencyScorer:
    def __init__(self, device: str = "cuda", viclip_pretrain_path: str = None):
        """
        Overall consistency reward based on VBench implementation.
        Uses ViCLIP to compute video-text cosine similarity.
        
        :param device: Device to run the model on
        :param viclip_pretrain_path: Path to ViCLIP pretrained weights
        """
        self.device = device
        
        cache_dir = os.environ.get("VBENCH_CACHE_DIR", os.path.expanduser("~/.cache/vbench"))
        
        if viclip_pretrain_path is None:
            viclip_pretrain_path = "/home/ma-user/work/xianyi/ckpts/OpenGVLab/ViCLIP/ViClip-InternVid-10M-FLT.pth"
        
        # Import ViCLIP from local third_party
        try:
            from .third_party.ViCLIP.viclip import ViCLIP
            from .third_party.ViCLIP.simple_tokenizer import SimpleTokenizer
        except ImportError:
            import sys
            sys.path.insert(0, os.path.dirname(__file__))
            from third_party.ViCLIP.viclip import ViCLIP
            from third_party.ViCLIP.simple_tokenizer import SimpleTokenizer
        
        tokenizer_path = "/home/ma-user/work/xianyi/ckpts/OpenGVLab/ViCLIP/bpe_simple_vocab_16e6.txt.gz"
        self.tokenizer = SimpleTokenizer(tokenizer_path)
        
        self.model = ViCLIP(
            tokenizer=self.tokenizer,
            pretrain=viclip_pretrain_path
        ).to(device)
        self.model.eval()
        
        # CLIP-style transform for ViCLIP (224x224)
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)
            ),
        ])
        
        self.num_frames = 8  # ViCLIP expects 8 frames

    def _sample_frames(self, frames: torch.Tensor, num_frames: int = 8) -> torch.Tensor:
        """
        Sample frames using 'middle' strategy (uniform sampling).
        
        :param frames: (N, C, H, W) tensor
        :param num_frames: Target number of frames
        :return: (num_frames, C, H, W) tensor
        """
        total = len(frames)
        if total >= num_frames:
            # Uniform sampling
            indices = np.linspace(0, total - 1, num_frames, dtype=int)
        else:
            # Repeat last frame to fill
            indices = list(range(total)) + [total - 1] * (num_frames - total)
        
        return frames[indices]

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

    @torch.no_grad()
    def __call__(
        self,
        images: Union[List[Image.Image], List[np.ndarray], np.ndarray],
        prompts: List[str],
    ) -> List[float]:
        """
        Calculate overall consistency reward (video-text similarity).
        
        :param images: List of images. For video: each element is np.ndarray (F,H,W,C).
                       For single image: treated as a 1-frame video.
        :param prompts: List of text prompts
        :return: List of reward scores (cosine similarity, typically in [0, 1])
        """
        assert len(images) == len(prompts), "Images and prompts must have the same length"
        
        frame_tensors = self._images_to_frame_tensors(images)
        rewards = []
        text_feature_dict = {}
        
        for frames, prompt in zip(frame_tensors, prompts):
            # Sample 8 frames for ViCLIP
            sampled = self._sample_frames(frames, self.num_frames)
            
            # Apply transform
            sampled_transformed = self.transform(sampled).to(self.device)
            
            # Encode video: ViCLIP expects (B, T, C, H, W)
            vid_input = sampled_transformed.unsqueeze(0)  # (1, 8, C, H, W)
            vid_feat = self.model.encode_vision(vid_input, test=True).float()
            vid_feat = F.normalize(vid_feat, dim=-1, p=2)
            
            # Encode text
            if prompt in text_feature_dict:
                text_feat = text_feature_dict[prompt]
            else:
                text_feat = self.model.encode_text(prompt).float()
                text_feat = F.normalize(text_feat, dim=-1, p=2)
                text_feature_dict[prompt] = text_feat
            
            # Compute cosine similarity
            score = (vid_feat @ text_feat.T).item()
            
            # Clamp to [0, 1] for reward
            score = max(0.0, min(1.0, score))
            rewards.append(score)
        
        return rewards


if __name__ == "__main__":
    scorer = VBenchOverallConsistencyScorer(device="cpu")
    frames = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(8)]
    video = np.stack(frames)
    reward = scorer([video], ["a cat playing with a ball"])
    print(f"Overall Consistency Reward: {reward}")
