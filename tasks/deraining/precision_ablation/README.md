# Precision Ablation

对比 FP32 / FP16 / BF16 三种训练精度对 GOUB (linear schedule) 去雨效果的影响。

FP16 使用 GradScaler 防止梯度下溢，BF16/FP32 不使用。

## 参数

| 参数 | 值 |
|------|-----|
| SDE | GOUB (schedule=linear) |
| Precision | fp32 / fp16 / bf16 |
| 迭代 | 200,000 |

## 启动

```bash
bash scripts/run_fp32.sh
bash scripts/run_fp16.sh
bash scripts/run_bf16.sh
```

或手动指定 GPU：

```bash
GPU=2 DATA_ROOT=source/DeRain-H bash scripts/run_bf16.sh
```

## 结果可视化

打开 `tasks/deraining/precision_ablation/plot.ipynb` 运行所有 cell 即可。
