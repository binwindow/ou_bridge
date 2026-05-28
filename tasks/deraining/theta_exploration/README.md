# GOUB Theta Schedule Exploration

对比 GOUB 三种 theta 调度策略对去雨效果的影响。

## 参数

| 参数 | 值 |
|------|-----|
| SDE | GOUB (λ²=30, T=100, ε=0.005) |
| Theta 调度 | cosine / linear / constant |

## 启动

```bash
bash scripts/run_cosine.sh   # theta = cosine schedule
bash scripts/run_linear.sh   # theta = linear schedule
bash scripts/run_constant.sh # theta = constant schedule
```

或手动指定 GPU：

```bash
GPU=2 DATA_ROOT=source/DeRain-H bash scripts/run_cosine.sh
```
