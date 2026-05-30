import json
import os
from typing import Optional

import torch


class CheckpointManager:
    """
    Manages checkpoint saving/loading with top-k PSNR tracking.

    Saves full training state (model, optimizer, scheduler, iteration)
    only at validation time. Maintains top-3 checkpoints by PSNR.
    """

    def __init__(self, ckpt_dir: str, topk: int = 3):
        self.ckpt_dir = ckpt_dir
        self.topk = topk
        os.makedirs(ckpt_dir, exist_ok=True)
        self._topk_info: list[dict] = self._load_topk_info()

    def _info_path(self) -> str:
        return os.path.join(self.ckpt_dir, "checkpoint_info.json")

    def _load_topk_info(self) -> list[dict]:
        path = self._info_path()
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return []

    def _save_topk_info(self):
        with open(self._info_path(), "w") as f:
            json.dump(self._topk_info, f, indent=2)

    def _ckpt_path(self, name: str) -> str:
        return os.path.join(self.ckpt_dir, name)

    def save(self, model, ema_model, optimizer, scheduler, iteration: int, metrics: Optional[dict] = None):
        """
        Save last.ckpt, and if metrics has 'psnr', manage top-k.
        Returns list of ckpt filenames saved.
        """
        saved = []

        if metrics is not None and "psnr" in metrics:
            psnr = metrics["psnr"]
            ckpt_name = f"iter_{iteration:06d}_psnr_{psnr:.3f}.ckpt"

            # Save new entry before registering, so crash won't leave
            # checkpoint_info.json pointing to a missing file.
            entry_path = self._ckpt_path(ckpt_name)
            if not os.path.exists(entry_path):
                self._save_checkpoint(entry_path, model, ema_model, optimizer, scheduler, iteration)
                saved.append(ckpt_name)

            entry = {
                "psnr": psnr,
                "ssim": metrics.get("ssim", None),
                "lpips": metrics.get("lpips", None),
                "iteration": iteration,
                "ckpt_name": ckpt_name,
            }
            self._topk_info.append(entry)
            self._topk_info.sort(key=lambda x: x["psnr"], reverse=True)

            # Evict entries beyond top-k
            while len(self._topk_info) > self.topk:
                removed = self._topk_info.pop()
                removed_path = self._ckpt_path(removed["ckpt_name"])
                if os.path.exists(removed_path):
                    os.remove(removed_path)

            self._save_topk_info()

        # Save last.ckpt (after topk_info is updated)
        last_path = self._ckpt_path("last.ckpt")
        self._save_checkpoint(last_path, model, ema_model, optimizer, scheduler, iteration)
        saved.append("last.ckpt")

        return saved

    def _save_checkpoint(self, path, model, ema_model, optimizer, scheduler, iteration):
        state = {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "ema_model_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "topk_info": self._topk_info,
        }
        torch.save(state, path)

    def load(self, path: str, model, ema_model, optimizer, scheduler, device):
        """Load checkpoint and return (iteration, topk_info)."""
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        ema_model.load_state_dict(state["ema_model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        self._topk_info = state.get("topk_info", [])
        self._save_topk_info()
        return state["iteration"], self._topk_info

    def has_checkpoint(self) -> bool:
        return os.path.exists(self._ckpt_path("last.ckpt"))

    @property
    def topk_info(self) -> list[dict]:
        return self._topk_info
