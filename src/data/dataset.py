import json
import os
import random

import numpy as np
import torch
import torch.utils.data as data

from . import transforms as util


class PairedImageDataset(data.Dataset):
    """
    Paired image dataset driven by a JSON manifest.

    JSON format:
        [{"input": "relative/path/to/lq.png", "target": "relative/path/to/gt.png"}, ...]

    Paths are resolved relative to `root_dir`.
    """

    def __init__(self, json_path, root_dir, patch_size=64, phase="train",
                 use_flip=True, use_rot=True):
        super().__init__()
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.phase = phase
        self.use_flip = use_flip
        self.use_rot = use_rot

        with open(json_path, "r") as f:
            self.pairs = json.load(f)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        lq_path = os.path.join(self.root_dir, pair["input"])
        gt_path = os.path.join(self.root_dir, pair["target"])

        img_lq = util.read_img(lq_path)
        img_gt = util.read_img(gt_path)

        if self.phase == "train":
            # random crop
            H, W, _ = img_lq.shape
            if H < self.patch_size or W < self.patch_size:
                raise RuntimeError(f"Image {lq_path} ({H}x{W}) smaller than patch_size {self.patch_size}")

            rnd_h = random.randint(0, H - self.patch_size)
            rnd_w = random.randint(0, W - self.patch_size)
            img_lq = img_lq[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            img_gt = img_gt[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # augmentation
            img_lq, img_gt = util.augment([img_lq, img_gt], self.use_flip, self.use_rot)

        # HWC BGR -> CHW RGB tensor
        img_lq = util.img2tensor(img_lq)
        img_gt = util.img2tensor(img_gt)

        return {"LQ": img_lq, "GT": img_gt, "LQ_path": lq_path, "GT_path": gt_path}
