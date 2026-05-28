# SDE Bridge Comparison

对比三种 SDE 桥模型对去雨效果的影响。

## 参数

| SDE | 核心参数 |
|-----|----------|
| GOUB | schedule=cosine, λ²=30, T=100 |
| VE-bridge | σ∈[0.002, 80], σ_data=0.5 |
| VP-bridge | t∈[0.0001, 1], β_d=2, β_min=0.1 |

## 启动

```bash
bash scripts/run_goub.sh   # GOUB
bash scripts/run_ve.sh     # VE-bridge
bash scripts/run_vp.sh     # VP-bridge
```

或手动指定 GPU：

```bash
GPU=2 DATA_ROOT=source/DeRain-H bash scripts/run_goub.sh
```
