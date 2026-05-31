# cuDNN Deterministic Ablation (FP32)

对比 `torch.backends.cudnn.benchmark = True`（默认，性能优化）与 `benchmark=False, deterministic=True`（可复现）对 GOUB (linear schedule) 去雨效果的影响。

两组均使用 ema=0.999, FP32 精度。

## 参数

| 参数 | 值 |
|------|-----|
| SDE | GOUB (schedule=linear) |
| EMA β | 0.999 |
| 精度 | FP32 (AMP disabled) |
| 迭代 | 200,000 |

## 启动

```bash
# 默认（benchmark=True）
bash tasks/deraining/cudnn_deterministic_ablation/scripts/run_default.sh

# 确定性模式（benchmark=False, deterministic=True）
bash tasks/deraining/cudnn_deterministic_ablation/scripts/run_deterministic.sh
```

## 观察

- 训练 loss 曲线和验证 PSNR/SSIM/LPIPS 是否一致
- step_time 差异
