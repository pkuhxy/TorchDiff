"""
OCR Reward Scorer — subprocess-isolated PaddleOCR.

PaddlePaddle initialises its own CUDA context upon `import paddle`, which
conflicts with PyTorch-managed CUDA in distributed (FSDP / NCCL) training.
To work around this we run PaddleOCR in a **separate process** that never
shares the PyTorch CUDA address space.

Communication:
    main process  —[request Queue]→  OCR worker process
    main process  ←[response Queue]—  OCR worker process

Each request is a dict:  {"id": int, "image": np.ndarray}
Each response is a dict: {"id": int, "text": str}
A sentinel {"id": -1} tells the worker to exit.
"""

import os
import sys
import multiprocessing as mp
import numpy as np
import torch
from Levenshtein import distance
from typing import List, Union, Tuple
from PIL import Image


# ---------------------------------------------------------------------------
# Worker function — runs in a child process with a clean CUDA environment
# ---------------------------------------------------------------------------

def _ocr_worker(req_queue: mp.Queue, resp_queue: mp.Queue):
    """Long-running OCR worker that lives in its own process."""
    # Completely prevent PaddlePaddle from touching CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["FLAGS_selected_gpus"] = ""
    os.environ["PADDLE_NO_GPU"] = "1"
    os.environ["FLAGS_use_cuda_managed_memory"] = "false"

    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=False,
            show_log=False,
        )
        print("[OCR Worker] PaddleOCR initialized successfully.", flush=True)
    except Exception as e:
        print(f"[OCR Worker] FATAL: Failed to init PaddleOCR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Drain the queue so the main process doesn't hang
        while True:
            msg = req_queue.get()
            if msg["id"] == -1:
                break
            resp_queue.put({"id": msg["id"], "text": ""})
        return

    while True:
        msg = req_queue.get()
        if msg["id"] == -1:  # shutdown sentinel
            break
        try:
            result = ocr.ocr(msg["image"], cls=False)
            text = ''.join(
                [res[1][0] if res[1][1] > 0 else '' for res in result[0]]
            ) if result and result[0] else ''
        except Exception as e:
            print(f"[OCR Worker] ocr.ocr failed: {e}", flush=True)
            text = ""
        resp_queue.put({"id": msg["id"], "text": text})


class _OcrProxy:
    """
    Proxy that forwards OCR requests to a background subprocess.
    Safe to create from a PyTorch distributed process.
    """

    def __init__(self):
        ctx = mp.get_context("spawn")  # "spawn" gives a clean process
        self._req_q = ctx.Queue()
        self._resp_q = ctx.Queue()
        self._proc = ctx.Process(
            target=_ocr_worker,
            args=(self._req_q, self._resp_q),
            daemon=True,
        )
        self._proc.start()
        self._counter = 0
        print(f"[OCR Proxy] Worker subprocess started (pid={self._proc.pid}).", flush=True)

    def ocr(self, image: np.ndarray) -> str:
        """Send one image and get back the recognised text (blocking)."""
        rid = self._counter
        self._counter += 1
        self._req_q.put({"id": rid, "image": image})
        resp = self._resp_q.get()  # blocks until the worker replies
        assert resp["id"] == rid, f"OCR response id mismatch: expected {rid}, got {resp['id']}"
        return resp["text"]

    def ocr_batch(self, images: List[np.ndarray]) -> List[str]:
        """Send a batch of images and get back results (blocking, sequential)."""
        results = []
        for img in images:
            results.append(self.ocr(img))
        return results

    def shutdown(self):
        """Gracefully stop the worker."""
        try:
            self._req_q.put({"id": -1})
            self._proc.join(timeout=10)
        except Exception:
            pass

    def __del__(self):
        self.shutdown()


# ---------------------------------------------------------------------------
# Scorer classes — public API (unchanged signatures)
# ---------------------------------------------------------------------------

def _extract_quoted_text(prompt: str) -> str:
    parts = prompt.split('"')
    if len(parts) >= 2:
        return parts[1]
    for q in ['\u201c', '\u201d', "'", '\u2018', '\u2019']:
        parts = prompt.split(q)
        if len(parts) >= 2:
            return parts[1]
    return prompt


class OcrScorer:
    def __init__(self, use_gpu: bool = False):
        """OCR reward calculator (subprocess-isolated PaddleOCR)."""
        self.proxy = _OcrProxy()

    @torch.no_grad()
    def __call__(
        self,
        images: Union[List[Image.Image], List[np.ndarray]],
        prompts: List[str],
    ) -> list:
        prompts = [_extract_quoted_text(p) for p in prompts]
        assert len(images) == len(prompts), "Images and prompts must have the same length"

        rewards = []
        for img, prompt in zip(images, prompts):
            if isinstance(img, Image.Image):
                img = np.array(img)
            try:
                recognized_text = self.proxy.ocr(img)
                recognized_text = recognized_text.replace(' ', '').lower()
                prompt_clean = prompt.replace(' ', '').lower()
                if prompt_clean in recognized_text:
                    dist_val = 0
                else:
                    dist_val = distance(recognized_text, prompt_clean)
                if dist_val > len(prompt_clean):
                    dist_val = len(prompt_clean)
            except Exception as e:
                print(f"OCR processing failed: {e}")
                dist_val = len(prompt)
            reward = 1 - dist_val / max(len(prompt), 1)
            rewards.append(reward)
        return rewards


class OcrScorer_video_or_image:
    def __init__(self, use_gpu: bool = False):
        """OCR reward calculator for video/image (subprocess-isolated PaddleOCR)."""
        self.proxy = _OcrProxy()
        self.frame_interval = 4

    @torch.no_grad()
    def __call__(
        self,
        images: Union[List[Image.Image], List[np.ndarray]],
        prompts: List[str],
    ) -> List[float]:
        prompts = [_extract_quoted_text(p) for p in prompts]
        assert len(images) == len(prompts), "Mismatch between images and prompts."

        rewards = []
        for img, prompt in zip(images, prompts):
            prompt_clean = prompt.replace(' ', '').lower()
            frame_rewards = []

            # Handle video: shape (F, H, W, C)
            if isinstance(img, np.ndarray) and img.ndim == 4:
                sampled_frames = img[::self.frame_interval]
            else:
                sampled_frames = [img]

            for frame in sampled_frames:
                if isinstance(frame, Image.Image):
                    frame = np.array(frame)
                try:
                    text = self.proxy.ocr(frame)
                    text = text.replace(' ', '').lower()
                    dist_val = distance(text, prompt_clean)
                    dist_val = min(dist_val, len(prompt_clean))
                except Exception as e:
                    print(f"OCR failed on frame: {e}")
                    dist_val = len(prompt_clean)

                reward = 1 - dist_val / max(len(prompt_clean), 1)
                if reward > 0:
                    frame_rewards.append(reward)

            if frame_rewards:
                rewards.append(sum(frame_rewards) / len(frame_rewards))
            else:
                rewards.append(0.0)

        return rewards


if __name__ == "__main__":
    example_image_path = "media_images_eval_images_499_ef42de47b8ec98892954.jpg"
    example_image = Image.open(example_image_path)
    example_prompt = 'New York Skyline with "Hello World" written with fireworks on the sky'
    scorer = OcrScorer(use_gpu=False)
    reward = scorer([example_image], [example_prompt])
    print(f"OCR Reward: {reward}")
