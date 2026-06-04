"""
Thin entry point for training.

Usage:
    python tasks/train.py --dataset derainh --exp_name derain_cosine --schedule cosine
    python tasks/train.py --dataset sccid --exp_name precision_fp32_bs16 --precision fp32
    python tasks/train.py --exp_name derain_cosine --resume
"""
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ExperimentConfig
from src.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="GOUB Training")
    parser.add_argument("--exp_name", type=str, default="experiment", help="Experiment name")
    parser.add_argument("--sde_type", type=str, default="goub", choices=["goub", "ve", "vp"])
    parser.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear", "constant"])
    parser.add_argument("--total_iterations", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--ema_beta", type=float, default=0.9999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_steps_sampling", type=int, default=100)
    parser.add_argument("--num_val", type=int, default=15)
    parser.add_argument("--dataset", type=str, default="derainh")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--output_root", type=str, default="outputs", help="Root directory for outputs")
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"],
                        help="Training precision (fp32=fp32, fp16=fp16+scalar, bf16=bf16)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.resume:
        # Resume: load config from saved config.json, override runtime params
        config_path = os.path.join(args.output_root, args.exp_name, "log", "config.json")
        config = ExperimentConfig.from_json(config_path)
        config.exp_name = args.exp_name
        config.output_root = args.output_root
        config.train.gpu = args.gpu
        config.data.data_root = args.data_root or config.data.data_root
        config.sde.num_steps_sampling = args.num_steps_sampling
        config.train.num_val = args.num_val
        config.train.ema_beta = args.ema_beta
    else:
        config = ExperimentConfig(exp_name=args.exp_name)
        config.output_root = args.output_root
        config.sde.sde_type = args.sde_type
        config.sde.schedule = args.schedule
        config.train.total_iterations = args.total_iterations
        config.train.batch_size = args.batch_size
        config.train.patch_size = args.patch_size
        config.train.lr = args.lr
        config.train.gpu = args.gpu
        config.train.ema_beta = args.ema_beta
        config.train.seed = args.seed
        config.sde.num_steps_sampling = args.num_steps_sampling
        config.train.num_val = args.num_val
        config.data.dataset = args.dataset
        config.data.data_root = args.data_root

    config.train.precision = args.precision

    trainer = Trainer(config)

    if args.resume:
        trainer.resume()

    trainer.train()


if __name__ == "__main__":
    main()
