"""Smoke test: verify the full pipeline with a tiny model. Run before training."""
import sys
import tempfile
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.config import ExperimentConfig
from src.sde import GOUB
from src.model import DenoisingModel
from src.trainer import set_seed

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Use small config to keep smoke test fast on any machine
config = ExperimentConfig(exp_name="smoke_test")
config.network.nf = 16
config.network.depth = 2

# 1. SDE
print("\n--- SDE ---")
sde = GOUB(lambda_square=30, T=100, schedule='cosine', eps=0.005, device=device)
x0 = torch.randn(1, 3, 64, 64).to(device)
mu = torch.randn(1, 3, 64, 64).to(device)
timesteps, states = sde.generate_random_states(x0, mu)
print(f"  states shape: {states.shape}")
print(f"  thetas range: [{sde.thetas.min():.4f}, {sde.thetas.max():.4f}]")

# 2. Model forward
print("\n--- Model ---")
model = DenoisingModel(config, device)
print(f"  total params: {model.get_parameter_info()['total_params']:,}")

model.feed_data(states, mu, x0)
noise = model.model(states, mu, timesteps.squeeze())
print(f"  noise shape: {noise.shape}")

# 3. Optimize step
print("\n--- Optimize ---")
optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
optimizer.zero_grad()
loss = model.optimize_parameters(0, timesteps, sde)
loss.backward()
optimizer.step()
model.ema.update(model.model)
print(f"  loss: {loss.item():.6f}")

# 4. Inference (2-step reverse for speed)
print("\n--- Inference ---")
model.model.eval()
with torch.no_grad():
    output = sde.reverse_sde(states[:1], mu[:1], model.model, T=2)
print(f"  output shape: {output.shape}")

# 5. Checkpoint save/load
print("\n--- Checkpoint ---")
from src.ckpt import CheckpointManager
with tempfile.TemporaryDirectory() as tmpdir:
    mgr = CheckpointManager(tmpdir)
    saved = mgr.save(
        model=model.model, ema_model=model.ema,
        optimizer=optimizer, scheduler=scheduler,
        iteration=10, metrics={"psnr": 25.0, "ssim": 0.85, "lpips": 0.15},
    )
    print(f"  saved: {saved}")
    assert mgr.has_checkpoint()

    # Resume
    resumed_iter, topk = mgr.load(
        os.path.join(tmpdir, "last.ckpt"), model=model.model, ema_model=model.ema,
        optimizer=optimizer, scheduler=scheduler, device=device,
    )
    print(f"  resume OK, iter={resumed_iter}, topk_count={len(topk)}")

# 6. Config roundtrip
print("\n--- Config roundtrip ---")
config2 = ExperimentConfig.from_dict(config.to_dict())
assert config2.sde.schedule == 'cosine'
assert config2.network.nf == 16
print("  OK")

print("\n=== All smoke tests passed ===")
