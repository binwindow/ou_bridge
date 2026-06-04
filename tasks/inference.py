"""
Inference script: load a checkpoint and run inference on test set.

Usage:
    python tasks/inference.py --exp_name precision_fp32_bs16 --output_root outputs/derainh [--ckpt last] [--gpu 2]
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ExperimentConfig
from src.sde import create_sde
from src.model import DenoisingModel
from src.data import create_dataloader
from src.logging.metrics import tensor2img, compute_batch_metrics
from src.trainer import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="outputs", help="Root directory for outputs")
    parser.add_argument("--ckpt", type=str, default="last", help="ckpt name or path, e.g. 'last' or 'iter_001000_psnr_25.000.ckpt'")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load experiment config
    exp_dir = os.path.join(args.output_root, args.exp_name)
    config = ExperimentConfig.from_json(os.path.join(exp_dir, "log", "config.json"))
    config.output_root = args.output_root
    if args.data_root:
        config.data.data_root = args.data_root
    set_seed(config.train.seed)

    # Build model
    model = DenoisingModel(config, device)
    sc = config.sde
    sde_kwargs = dict(device=device)
    if sc.sde_type == "goub":
        sde_kwargs.update(lambda_square=sc.lambda_square, T=sc.T, schedule=sc.schedule, eps=sc.eps)
    elif sc.sde_type == "ve":
        sde_kwargs.update(sigma_max=sc.sigma_max, sigma_min=sc.sigma_min, sigma_data=sc.sigma_data,
                          num_steps=sc.num_steps_sampling, rho=sc.rho)
    elif sc.sde_type == "vp":
        sde_kwargs.update(sigma_max=sc.sigma_max, sigma_min=sc.sigma_min, sigma_data=sc.sigma_data,
                          beta_d=sc.beta_d, beta_min=sc.beta_min, num_steps=sc.num_steps_sampling)
    sde = create_sde(sc.sde_type, **sde_kwargs)

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

    # Data — dataset type and preprocessing read from saved config
    loader = create_dataloader(
        config.data.dataset,
        batch_size=args.batch_size,
        num_workers=1,
        phase="val",
        root_dir=config.data.data_root,
        patch_size=config.train.patch_size,
        use_flip=False,
        use_rot=False,
    )

    # Inference
    test_dir = os.path.join(exp_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    all_metrics = {"psnr": [], "ssim": [], "lpips": []}

    for idx, batch in enumerate(tqdm(loader, desc="Inference", ncols=100, dynamic_ncols=False)):
        LQ = batch["LQ"]
        GT = batch["GT"]
        lq_path = batch["LQ_path"][0]

        with torch.no_grad():
            output = sde.sample(model.ema.ema_model, LQ.to(device), LQ.to(device))

        # Debug: print tensor stats for first image
        if idx == 0:
            print(f"  LQ    range: [{LQ.min():.4f}, {LQ.max():.4f}] mean={LQ.mean():.4f}")
            print(f"  GT    range: [{GT.min():.4f}, {GT.max():.4f}] mean={GT.mean():.4f}")
            print(f"  output range: [{output.min():.4f}, {output.max():.4f}] mean={output.mean():.4f}")

        # Clamp and save
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
