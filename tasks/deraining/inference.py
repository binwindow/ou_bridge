"""
Inference script: load a checkpoint and run deraining on test set.

Usage:
    python tasks/deraining/inference.py --exp_name derain_cosine [--ckpt last] [--gpu 2]
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import ExperimentConfig
from src.sde import GOUB
from src.model import DenoisingModel
from src.data import create_dataloader
from src.logging.metrics import tensor2img, compute_batch_metrics
from src.trainer import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="last", help="ckpt name or path, e.g. 'last' or 'topk_1_xxx.ckpt'")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--data_root", type=str, default="source/DeRain-H")
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load experiment config
    exp_dir = f"outputs/{args.exp_name}"
    config = ExperimentConfig.from_json(os.path.join(exp_dir, "log", "config.json"))
    config.data.data_root = args.data_root
    set_seed(config.train.seed)

    # Build model
    model = DenoisingModel(config, device)
    sde = GOUB(
        lambda_square=config.sde.lambda_square,
        T=config.sde.T,
        schedule=config.sde.schedule,
        eps=config.sde.eps,
        device=device,
    )

    # Load checkpoint
    ckpt_dir = os.path.join(exp_dir, "ckpt")
    if args.ckpt == "last":
        ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
    elif os.path.exists(args.ckpt):
        ckpt_path = args.ckpt
    else:
        ckpt_path = os.path.join(ckpt_dir, args.ckpt)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.model.load_state_dict(state["model_state_dict"])
    model.ema.load_state_dict(state["ema_model_state_dict"])
    print(f"Loaded checkpoint: {ckpt_path} (iter {state['iteration']})")

    # Use EMA model for inference
    model.ema.ema_model.eval()

    # Data
    root = args.data_root
    val_json = os.path.join(root, "test.json") if os.path.exists(os.path.join(root, "test.json")) else os.path.join(root, "val.json")
    loader = create_dataloader(
        config.data.dataset,
        batch_size=args.batch_size,
        num_workers=1,
        phase="val",
        json_path=val_json,
        root_dir=root,
        patch_size=config.train.patch_size,
        use_flip=False,
        use_rot=False,
    )

    # Inference
    test_dir = os.path.join(exp_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    all_metrics = {"psnr": [], "ssim": [], "lpips": []}

    for idx, batch in enumerate(tqdm(loader, desc="Inference")):
        LQ = batch["LQ"]
        GT = batch["GT"]
        lq_path = batch["LQ_path"][0]

        with torch.no_grad():
            output = sde.reverse_sde(LQ.to(device), LQ.to(device), model.ema.ema_model, T=-1, save_states=False)

        # Debug: print tensor stats for first image
        if idx == 0:
            print(f"  LQ    range: [{LQ.min():.4f}, {LQ.max():.4f}] mean={LQ.mean():.4f}")
            print(f"  GT    range: [{GT.min():.4f}, {GT.max():.4f}] mean={GT.mean():.4f}")
            print(f"  output range: [{output.min():.4f}, {output.max():.4f}] mean={output.mean():.4f}")

        # Clamp and save
        # Use the output directly (it's already the reverse SDE result)
        out_np = tensor2img(output[0])
        save_path = os.path.join(test_dir, os.path.basename(lq_path))
        cv2.imwrite(save_path, out_np)

        # Metrics (only on first, to speed up)
        metrics = compute_batch_metrics(output, GT, device)
        for k in all_metrics:
            all_metrics[k].append(metrics[k])

    # Report
    print(f"\n--- Test Results ({args.exp_name}) ---")
    print(f"  PSNR:  {np.mean(all_metrics['psnr']):.3f} dB")
    print(f"  SSIM:  {np.mean(all_metrics['ssim']):.4f}")
    print(f"  LPIPS: {np.mean(all_metrics['lpips']):.4f}")
    print(f"  Images saved to: {test_dir}")


if __name__ == "__main__":
    main()
