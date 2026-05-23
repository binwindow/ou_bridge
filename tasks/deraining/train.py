"""
Thin entry point for deraining training.

Usage:
    python tasks/deraining/train.py --exp_name derain_cosine --schedule cosine
    python tasks/deraining/train.py --exp_name derain_cosine --resume
"""
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import ExperimentConfig
from src.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="GOUB Deraining Training")
    parser.add_argument("--exp_name", type=str, default="derain_cosine", help="Experiment name")
    parser.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear", "constant"])
    parser.add_argument("--total_iterations", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_val", type=int, default=15)
    parser.add_argument("--dataset", type=str, default="rain100h")
    parser.add_argument("--dataroot_GT", type=str, default="")
    parser.add_argument("--dataroot_LQ", type=str, default="")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    config = ExperimentConfig(
        exp_name=args.exp_name,
    )
    config.sde.schedule = args.schedule
    config.train.total_iterations = args.total_iterations
    config.train.batch_size = args.batch_size
    config.train.patch_size = args.patch_size
    config.train.lr = args.lr
    config.train.seed = args.seed
    config.train.num_val = args.num_val
    config.data.dataset = args.dataset
    config.data.dataroot_GT = args.dataroot_GT
    config.data.dataroot_LQ = args.dataroot_LQ

    trainer = Trainer(config)

    if args.resume:
        trainer.resume()

    trainer.train()


if __name__ == "__main__":
    main()
