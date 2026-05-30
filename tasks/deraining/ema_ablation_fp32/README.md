# EMA Decay Ablation (FP32)

对比 EMA 衰减系数（FP32 精度）对 GOUB (linear schedule) 去雨效果的影响。

## 参数

| 参数 | 值 |
|------|-----|
| SDE | GOUB (schedule=linear) |
| EMA β | 0.99 / 0.999 / 0.9999 |
| 精度 | FP32 (AMP disabled) |
| 迭代 | 200,000 |

## 启动

```bash
bash scripts/run_ema_099_fp32.sh
bash scripts/run_ema_0999_fp32.sh
bash scripts/run_ema_09999_fp32.sh
```
