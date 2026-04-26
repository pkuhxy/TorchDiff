"""
VBench Dynamic Degree Reward
Based on RAFT optical flow model.
Computes optical flow between consecutive frames, uses the top-5% magnitude
as the motion score. Returns a normalized dynamic degree score.

Input: list of multi-frame images (as video frames) + prompts
Output: list of float rewards
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Union
from PIL import Image
from easydict import EasyDict as edict


class VBenchDynamicDegreeScorer:
    def __init__(self, device: str = "cuda", raft_model_path: str = None):
        """
        Dynamic degree reward based on VBench implementation.
        Uses RAFT optical flow to measure motion intensity.
        
        :param device: Device to run the model on
        :param raft_model_path: Path to RAFT model weights (raft-things.pth)
        """
        self.device = device
        
        if raft_model_path is None:
            cache_dir = os.environ.get("VBENCH_CACHE_DIR", os.path.expanduser("~/.cache/vbench"))
            raft_model_path = "/home/ma-user/work/xianyi/ckpts/iic/cv_dut-raft_video-stabilization_base/ckpt/raft-things.pth"
        
        # Import RAFT from local third_party
        try:
            from .third_party.RAFT.core.raft import RAFT
            from .third_party.RAFT.core.utils_core.utils import InputPadder
        except ImportError:
            import sys
            sys.path.insert(0, os.path.dirname(__file__))
            from third_party.RAFT.core.raft import RAFT
            from third_party.RAFT.core.utils_core.utils import InputPadder
        self.InputPadder = InputPadder
        
        args = edict({
            "model": raft_model_path,
            "small": False,
            "mixed_precision": False,
            "alternate_corr": False
        })
        self.model = RAFT(args)
        ckpt = torch.load(args.model, map_location="cpu")
        new_ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        self.model.load_state_dict(new_ckpt)
        self.model.to(device)
        self.model.eval()

    def _get_flow_score(self, flow: torch.Tensor) -> float:
        """
        Compute motion score from optical flow.
        Takes the mean of the top 5% flow magnitudes.
        """
        flo = flow[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)
        u = flo[:, :, 0]
        v = flo[:, :, 1]
        rad = np.sqrt(np.square(u) + np.square(v))
        
        h, w = rad.shape
        rad_flat = rad.flatten()
        cut_index = int(h * w * 0.05)
        if cut_index == 0:
            cut_index = 1
        
        max_rad = np.mean(abs(np.sort(-rad_flat))[:cut_index])
        return max_rad.item()

    def _set_params(self, frame_shape, count):
        """Set motion threshold and count threshold based on resolution and frame count."""
        scale = min(frame_shape[-2:])
        self.params = {
            "thres": 6.0 * (scale / 256.0),
            "count_num": round(4 * (count / 16.0))
        }

    def _check_move(self, score_list: List[float]) -> bool:
        """Check if enough frames exceed the motion threshold."""
        thres = self.params["thres"]
        count_num = self.params["count_num"]
        count = 0
        for score in score_list:
            if score > thres:
                count += 1
            if count >= count_num:
                return True
        return False

    def _compute_dynamic_score(self, frames: torch.Tensor) -> float:
        """
        Compute a continuous dynamic degree score from multiple frames.
        
        :param frames: Tensor of shape (N, C, H, W), pixel values in [0, 255]
        :return: Normalized dynamic degree score in [0, 1]
        """
        if len(frames) < 2:
            return 0.0
        
        self._set_params(frames[0].shape, len(frames))
        
        flow_scores = []
        for i in range(len(frames) - 1):
            image1 = frames[i:i+1].to(self.device)
            image2 = frames[i+1:i+2].to(self.device)
            
            padder = self.InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            
            _, flow_up = self.model(image1, image2, iters=20, test_mode=True)
            score = self._get_flow_score(flow_up)
            flow_scores.append(score)
        
        # Binary: does it move? (VBench original)
        has_motion = self._check_move(flow_scores)
        
        # Also compute a continuous score: mean flow magnitude normalized
        if flow_scores:
            mean_flow = np.mean(flow_scores)
            thres = self.params["thres"]
            # Normalize: score = min(mean_flow / (2 * thres), 1.0) for a softer signal
            # But also give 1.0 if clearly moving
            continuous_score = min(mean_flow / (2.0 * thres), 1.0) if thres > 0 else 0.0
        else:
            continuous_score = 0.0
        
        # Combine: if clearly moving, return at least 0.5
        if has_motion:
            return max(continuous_score, 0.5)
        else:
            return continuous_score

    def _images_to_frame_tensors(
        self, images: Union[List[Image.Image], List[np.ndarray], np.ndarray]
    ) -> List[torch.Tensor]:
        """
        Convert images to list of frame tensors.
        Each element: (N, C, H, W) in [0, 255] float range (RAFT expects this).
        
        For single-frame inputs, the dynamic degree will be 0.
        For multi-frame inputs (video), compute motion across frames.
        """
        result = []
        if isinstance(images, np.ndarray):
            if images.ndim == 4:  # (B, H, W, C)
                for i in range(images.shape[0]):
                    t = torch.from_numpy(images[i]).permute(2, 0, 1).float().unsqueeze(0)
                    result.append(t)
            elif images.ndim == 5:  # (B, F, H, W, C)
                for i in range(images.shape[0]):
                    t = torch.from_numpy(images[i]).permute(0, 3, 1, 2).float()
                    result.append(t)
        else:
            for img in images:
                if isinstance(img, Image.Image):
                    arr = np.array(img)
                    t = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0)
                    result.append(t)
                elif isinstance(img, np.ndarray):
                    if img.ndim == 3:
                        t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
                        result.append(t)
                    elif img.ndim == 4:
                        t = torch.from_numpy(img).permute(0, 3, 1, 2).float()
                        result.append(t)
                elif isinstance(img, torch.Tensor):
                    if img.ndim == 3:
                        result.append(img.float().unsqueeze(0))
                    elif img.ndim == 4:
                        result.append(img.float())
        return result

    @torch.no_grad()
    def __call__(
        self,
        images: Union[List[Image.Image], List[np.ndarray], np.ndarray],
        prompts: List[str],
    ) -> List[float]:
        """
        Calculate dynamic degree reward.
        
        :param images: List of images. For video: each element is np.ndarray (F,H,W,C)
                       For single image: returns 0.0 (no motion).
        :param prompts: List of text prompts (not used, kept for interface consistency)
        :return: List of reward scores in [0, 1]
        """
        assert len(images) == len(prompts), "Images and prompts must have the same length"
        
        frame_tensors = self._images_to_frame_tensors(images)
        rewards = []
        
        for frames in frame_tensors:
            # frames: (N, C, H, W) in [0, 255] float
            score = self._compute_dynamic_score(frames)
            rewards.append(score)
        
        return rewards


if __name__ == "__main__":
    scorer = VBenchDynamicDegreeScorer(device="cpu")
    # Simulate a 4-frame video with some motion
    frames = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(4)]
    video = np.stack(frames)  # (4, 256, 256, 3)
    reward = scorer([video], ["a person walking"])
    print(f"Dynamic Degree Reward: {reward}")
