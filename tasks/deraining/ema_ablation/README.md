# EMA Decay Ablation

对比 EMA 衰减系数对 GOUB (linear schedule) 去雨效果的影响。

## 参数

| 参数 | 值 |
|------|-----|
| SDE | GOUB (schedule=linear) |
| EMA β | 0.99 / 0.999 / 0.9999 |
| 迭代 | 200,000 |

## 启动

```bash
bash scripts/run_ema_099.sh
bash scripts/run_ema_0999.sh
bash scripts/run_ema_09999.sh
```
