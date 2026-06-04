import json
import os
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data

from . import transforms as util


class PairedImageDataset(data.Dataset):
    """
    Paired image dataset driven by a JSON manifest.

    Default JSON format:
        [{"input": "relative/path/to/lq.png", "target": "relative/path/to/gt.png"}, ...]

    Paths are resolved relative to `root_dir`.

    Subclasses override `_parse_entry` to support different JSON key names
    and `_preprocess` to customize the image preprocessing pipeline.
    """

    def __init__(self, root_dir, phase="train", patch_size=64,
                 use_flip=True, use_rot=True):
        super().__init__()
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.phase = phase
        self.use_flip = use_flip
        self.use_rot = use_rot

        json_path = self._resolve_json_path(root_dir, phase)
        with open(json_path, "r") as f:
            self.pairs = json.load(f)

    @staticmethod
    def _resolve_json_path(root_dir, phase):
        """Map (root_dir, phase) to a JSON manifest path. Override in subclasses."""
        if phase == "val" and not os.path.exists(os.path.join(root_dir, "val.json")):
            return os.path.join(root_dir, "test.json")
        return os.path.join(root_dir, f"{phase}.json")

    def _parse_entry(self, entry):
        """Return (lq_path, gt_path) relative to root_dir. Override in subclasses."""
        return entry["input"], entry["target"]

    def _preprocess(self, img_lq, img_gt):
        """Preprocess image pair before tensor conversion. Override in subclasses."""
        H, W, _ = img_lq.shape
        if self.phase == "train":
            if H < self.patch_size or W < self.patch_size:
                raise RuntimeError(f"Image ({H}x{W}) smaller than patch_size {self.patch_size}")

            rnd_h = random.randint(0, H - self.patch_size)
            rnd_w = random.randint(0, W - self.patch_size)
            img_lq = img_lq[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            img_gt = img_gt[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            img_lq, img_gt = util.augment([img_lq, img_gt], self.use_flip, self.use_rot)
        else:
            h_start = (H - self.patch_size) // 2
            w_start = (W - self.patch_size) // 2
            img_lq = img_lq[h_start:h_start + self.patch_size, w_start:w_start + self.patch_size, :]
            img_gt = img_gt[h_start:h_start + self.patch_size, w_start:w_start + self.patch_size, :]
        return img_lq, img_gt

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        lq_rel, gt_rel = self._parse_entry(pair)
        lq_path = os.path.join(self.root_dir, lq_rel)
        gt_path = os.path.join(self.root_dir, gt_rel)

        img_lq = util.read_img(lq_path)
        img_gt = util.read_img(gt_path)

        img_lq, img_gt = self._preprocess(img_lq, img_gt)

        # HWC BGR -> CHW RGB tensor
        img_lq = util.img2tensor(img_lq)
        img_gt = util.img2tensor(img_gt)

        return {"LQ": img_lq, "GT": img_gt, "LQ_path": lq_path, "GT_path": gt_path}


class SCCIDDataset(PairedImageDataset):
    """
    S-CCID dataset: Chinese calligraphy ink-bleed restoration.

    JSON format:
        [{"reference": "path/to/ink.png", "ground_truth": "path/to/origin.png"}, ...]

    Uses Resize(64,64) instead of random patch cropping. No augmentation.
    """

    def __init__(self, root_dir, phase="train", patch_size=64, **kwargs):
        super().__init__(root_dir, phase=phase, patch_size=patch_size,
                         use_flip=False, use_rot=False)

    def _parse_entry(self, entry):
        return entry["reference"], entry["ground_truth"]

    def _preprocess(self, img_lq, img_gt):
        img_lq = cv2.resize(img_lq, (self.patch_size, self.patch_size),
                            interpolation=cv2.INTER_LINEAR)
        img_gt = cv2.resize(img_gt, (self.patch_size, self.patch_size),
                            interpolation=cv2.INTER_LINEAR)
        return img_lq, img_gt
