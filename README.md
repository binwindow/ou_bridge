# GOUB — Generalized Ornstein-Uhlenbeck Bridge

Theta 调度参数对比实验（deraining）。

## 环境安装

```bash
# 1. 创建环境
conda create -n goub python=3.10 -y
conda activate goub

# 2. 安装依赖
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 数据集

JSON 清单格式 `[{input: "rel/path/lq.png", target: "rel/path/gt.png"}]`，目录下需有 `train.json` 和 `test.json`（或 `val.json`）。

```bash
# 软链接数据集
ln -s /path/to/dataset source/<name>
```

## 实验

| 实验 | 路径 | 说明 |
|------|------|------|
| Theta 调度对比 | `tasks/deraining/theta_exploration/` | GOUB 三种 schedule (cosine/linear/constant) |
| SDE 桥模型对比 | `tasks/deraining/sde_comparison/` | GOUB / VE-bridge / VP-bridge |

每个实验目录含 `README.md` 和 `scripts/`，进入后阅读即可启动。

```bash
# Theta 对比
bash tasks/deraining/theta_exploration/scripts/run_cosine.sh

# SDE 对比
bash tasks/deraining/sde_comparison/scripts/run_goub.sh
```

或直接调用共用入口：

```bash
python tasks/deraining/train.py \
  --exp_name my_exp \
  --sde_type goub \
  --schedule cosine \
  --gpu 0 \
  --total_iterations 200000 \
  --data_root source/DeRain-H
```

## 输出结构

```
outputs/<exp_name>/
├── log/
│   ├── config.json           # 完整配置
│   ├── parameter.json        # 模型参数量
│   ├── train_metrics.jsonl   # 训练指标（每100iter）
│   ├── val_metrics.jsonl     # 验证指标（每次val）
│   └── train.log             # 终端日志
├── ckpt/
│   ├── last.ckpt             # 最新检查点
│   ├── topk_*.ckpt           # PSNR 前3
│   └── checkpoint_info.json
├── samples/                  # val 生成样本
├── plt_fig/                  # 实验可视化图
└── test/                     # 最终测试结果
```

## 恢复训练

```bash
python tasks/deraining/train.py --exp_name my_exp --resume --gpu 2
```

## 可视化

```bash
# Theta 调度对比
jupyter notebook tasks/deraining/theta_exploration/visualize.ipynb

# SDE 桥对比
jupyter notebook tasks/deraining/sde_comparison/visualize.ipynb
```
