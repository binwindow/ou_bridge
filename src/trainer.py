import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
from torchvision.utils import save_image
from tqdm import tqdm

from .config import ExperimentConfig
from .sde import create_sde
from .model import DenoisingModel
from .data import create_dataloader
from .logging import Logger, compute_batch_metrics
from .ckpt import CheckpointManager


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.use_ddp = config.train.use_ddp

        # GPU selection: CUDA_VISIBLE_DEVICES limits visible devices;
        # local_rank then indexes into the visible set (0 → first visible GPU).
        if not self.use_ddp:
            os.environ["CUDA_VISIBLE_DEVICES"] = config.train.gpu

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.local_rank = 0

        # DDP setup
        if self.use_ddp:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.distributed.init_process_group(backend="nccl")

        self.is_main = self.local_rank == 0

        set_seed(config.train.seed)

        # Output directories
        if self.is_main:
            os.makedirs(config.log_dir, exist_ok=True)
            os.makedirs(config.ckpt_dir, exist_ok=True)
            os.makedirs(config.samples_dir, exist_ok=True)
            os.makedirs(config.plt_dir, exist_ok=True)
            os.makedirs(config.test_dir, exist_ok=True)

        # Model
        self.model = DenoisingModel(config, self.device)

        # SDE
        sde_cfg = config.sde
        sde_kwargs = dict(device=self.device)
        if sde_cfg.sde_type == "goub":
            sde_kwargs.update(lambda_square=sde_cfg.lambda_square, T=sde_cfg.T,
                              schedule=sde_cfg.schedule, eps=sde_cfg.eps)
        elif sde_cfg.sde_type == "ve":
            sde_kwargs.update(sigma_max=sde_cfg.sigma_max, sigma_min=sde_cfg.sigma_min,
                              sigma_data=sde_cfg.sigma_data, num_steps=sde_cfg.num_steps_sampling,
                              rho=sde_cfg.rho)
        elif sde_cfg.sde_type == "vp":
            sde_kwargs.update(sigma_data=sde_cfg.sigma_data,
                              beta_d=sde_cfg.beta_d, beta_min=sde_cfg.beta_min,
                              num_steps=sde_cfg.num_steps_sampling)
        self.sde = create_sde(sde_cfg.sde_type, **sde_kwargs)

        # Data
        self._create_dataloaders()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
            betas=(config.train.beta1, config.train.beta2),
        )

        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.train.total_iterations,
            eta_min=config.train.min_lr,
        )

        # AMP
        self.use_amp = config.train.use_amp and self.device.type == "cuda"
        self.scaler = GradScaler("cuda") if self.use_amp and self.device.type == "cuda" else None

        # DDP wrap
        if self.use_ddp:
            self.model.model = DDP(self.model.model, device_ids=[self.local_rank])

        # Logging & Checkpoint (main process only)
        self.logger = Logger(config.log_dir) if self.is_main else None
        self.ckpt_mgr = CheckpointManager(config.ckpt_dir) if self.is_main else None

        # State
        self.current_iteration = 0
        self.best_psnr = 0.0

        # Compute val iterations
        self.val_iterations = self._compute_val_iterations()

        if self.is_main:
            self.logger.save_config(config.to_dict())
            self.logger.save_parameter_info(self.model.get_parameter_info())
            self._print_config_summary()

    def _create_dataloaders(self):
        cfg = self.config

        data_kwargs = dict(
            root_dir=cfg.data.data_root,
            patch_size=cfg.train.patch_size,
            use_flip=cfg.data.use_flip,
            use_rot=cfg.data.use_rot,
        )

        # TODO: DDP — wrap with DistributedSampler when use_ddp=True

        self.train_loader = create_dataloader(
            cfg.data.dataset,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.data.num_workers,
            phase="train",
            **data_kwargs,
        )

        self.val_loader = create_dataloader(
            cfg.data.dataset,
            batch_size=1,
            num_workers=1,
            phase="val",
            **data_kwargs,
        )

    def _compute_val_iterations(self):
        total = self.config.train.total_iterations
        num_val = self.config.train.num_val
        return [int(total * (i + 1) / num_val) for i in range(num_val)]

    def train(self):
        cfg = self.config
        self.model.model.train()

        pbar = tqdm(
            total=cfg.train.total_iterations,
            initial=self.current_iteration,
            desc=f"[{cfg.exp_name}]",
            disable=not self.is_main,
            ncols=80,
            dynamic_ncols=False,
        )

        data_iter = iter(self.train_loader)
        step_time_start = time.time()

        while self.current_iteration < cfg.train.total_iterations:
            # Fetch batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            LQ = batch["LQ"]
            GT = batch["GT"]

            # Forward diffusion
            timesteps, states = self.sde.generate_random_states(GT, LQ)
            self.model.feed_data(xt=states, LQ=LQ, GT=GT)

            # Optimize
            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast("cuda"):
                    loss = self.model.optimize_parameters(timesteps, self.sde)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self.model.optimize_parameters(timesteps, self.sde)
                loss.backward()
                self.optimizer.step()

            # EMA update every step
            self.model.ema.update(self.model.model)

            self.scheduler.step()
            self.current_iteration += 1

            step_time = time.time() - step_time_start
            loss_val = loss.item()
            current_lr = self.scheduler.get_last_lr()[0]

            # Log
            if self.is_main and self.current_iteration % cfg.train.log_every == 0:
                self.logger.log_train({
                    "iter": self.current_iteration,
                    "loss": loss_val,
                    "lr": current_lr,
                    "step_time": round(step_time, 4),
                })

                pbar.set_postfix({
                    "loss": f"{loss_val:.4f}",
                    "lr": f"{current_lr:.2e}",
                })

            pbar.update(1)
            step_time_start = time.time()

            # Validation: save ckpt first, then log (avoid duplicate on crash)
            if self.is_main and self.current_iteration in self.val_iterations:
                val_metrics, sample_images = self._validate()
                self._save_checkpoint(val_metrics)
                self._log_and_save_samples(val_metrics, sample_images)

        pbar.close()

        if self.is_main:
            self.logger.close()

    def _validate(self) -> tuple[dict, list]:
        """Run inference on val set. Returns (metrics_dict, sample_images_list) without logging."""
        self.model.model.eval()

        all_metrics = {"psnr": [], "ssim": [], "lpips": []}
        sample_images = []

        val_pbar = tqdm(self.val_loader, desc="  val", ncols=80, dynamic_ncols=False, leave=False)

        for val_idx, batch in enumerate(val_pbar):
            LQ = batch["LQ"]
            GT = batch["GT"]

            self.model.feed_data(xt=LQ, LQ=LQ, GT=GT)
            output = self.model.infer(self.sde, use_ema=True)

            metrics = compute_batch_metrics(output, GT, self.device)
            for k in all_metrics:
                all_metrics[k].append(metrics[k])

            if val_idx < 4:
                sample_images.append({
                    "lq": LQ[0].cpu(),
                    "gt": GT[0].cpu(),
                    "output": output[0].cpu(),
                })

        self.model.model.train()

        avg_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}
        avg_metrics["iter"] = self.current_iteration
        avg_metrics["lr"] = self.scheduler.get_last_lr()[0]

        return avg_metrics, sample_images

    def _print_config_summary(self):
        cfg = self.config
        model_info = self.model.get_parameter_info()
        total_params = model_info["total_params"]
        train_size = len(self.train_loader.dataset)
        val_size = len(self.val_loader.dataset)
        amp_status = "AMP" if cfg.train.use_amp else "FP32"
        gpu_str = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        device_str = f"CUDA:{gpu_str} ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"

        sc = cfg.sde
        if sc.sde_type == "goub":
            sde_desc = f"GOUB | schedule={sc.schedule} | λ²={sc.lambda_square} | T={sc.T} | ε={sc.eps}"
        elif sc.sde_type == "ve":
            sde_desc = f"VE-bridge | σ∈[{sc.sigma_min},{sc.sigma_max}] | σ_data={sc.sigma_data}"
        elif sc.sde_type == "vp":
            sde_desc = f"VP-bridge | β_d={sc.beta_d} | β_min={sc.beta_min}"

        lines = [
            f"── {cfg.exp_name} " + "─" * (60 - len(cfg.exp_name)),
            f"  Model:    {cfg.network.architecture} | {total_params:,} params",
            f"  Data:     {cfg.data.data_root} | train: {train_size} | val: {val_size}",
            f"  Training: {cfg.train.total_iterations} iters | batch: {cfg.train.batch_size} | patch: {cfg.train.patch_size} | {amp_status}",
            f"  Optim:    {cfg.train.optimizer} | lr: {cfg.train.lr} → {cfg.train.min_lr} (CosineAnnealingLR)",
            f"  EMA:      β={cfg.train.ema_beta}, every {cfg.train.ema_update_every} step",
            f"  SDE:      {sc.sde_type} | {sde_desc}",
            f"  Device:   {device_str}",
            f"  Output:   outputs/{cfg.exp_name}/",
            "─" * 60,
        ]
        self.logger.info("\n" + "\n".join(lines))

    def _log_and_save_samples(self, avg_metrics: dict, sample_images: list):
        """Log val metrics and save sample grid images. Called AFTER checkpoint save."""
        if not self.logger:
            return

        self.logger.log_val(avg_metrics)
        self.logger.info(
            f"[Val] iter={self.current_iteration} | "
            f"PSNR: {avg_metrics['psnr']:.3f} | "
            f"SSIM: {avg_metrics['ssim']:.3f} | "
            f"LPIPS: {avg_metrics['lpips']:.4f} | "
            f"Best PSNR: {self.best_psnr:.3f}"
        )

        if avg_metrics["psnr"] > self.best_psnr:
            self.best_psnr = avg_metrics["psnr"]

        sample_dir = os.path.join(self.config.samples_dir, f"iter_{self.current_iteration:06d}")
        os.makedirs(sample_dir, exist_ok=True)
        for i, imgs in enumerate(sample_images):
            grid = torch.cat([
                torch.clamp(imgs["lq"], 0, 1),
                torch.clamp(imgs["gt"], 0, 1),
                torch.clamp(imgs["output"], 0, 1),
            ], dim=2)
            save_image(grid, os.path.join(sample_dir, f"sample_{i}_lq_gt_output.png"), normalize=False)

    def _save_checkpoint(self, metrics: dict):
        if self.ckpt_mgr:
            saved = self.ckpt_mgr.save(
                model=self.model.model,
                ema_model=self.model.ema,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                iteration=self.current_iteration,
                metrics=metrics,
            )
            if self.logger:
                self.logger.info(f"Checkpoint saved: {saved}")

    def resume(self):
        """Resume from last.ckpt."""
        ckpt_path = os.path.join(self.config.ckpt_dir, "last.ckpt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

        self.current_iteration, topk_info = self.ckpt_mgr.load(
            ckpt_path,
            model=self.model.model,
            ema_model=self.model.ema,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )

        if topk_info:
            self.best_psnr = max(e["psnr"] for e in topk_info)

        if self.logger:
            self.logger.info(f"Resumed from iteration {self.current_iteration}, best PSNR: {self.best_psnr:.3f}")
