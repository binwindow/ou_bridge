# GOUB — Generalized Ornstein-Uhlenbeck Bridge

Theta 调度参数对比实验（deraining）。

## 环境安装

```bash
# 1. 创建环境
conda create -n goub python=3.10 -y
conda activate goub

# 2. 安装依赖
pip install -r requirements.txt
```

## 数据集

JSON 清单格式 `[{input: "rel/path/lq.png", target: "rel/path/gt.png"}]`，目录下需有 `train.json` 和 `test.json`（或 `val.json`）。

```bash
# 软链接数据集
ln -s /path/to/dataset source/<name>
```

## 启动实验

```bash
bash scripts/run_cosine.sh
bash scripts/run_linear.sh
bash scripts/run_constant.sh

# 或直接调用
python tasks/deraining/train.py \
  --exp_name derain_cosine \
  --schedule cosine \
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
python tasks/deraining/train.py --exp_name derain_cosine --resume
```

## 可视化

```bash
jupyter notebook visualization.ipynb
```

- `visualize_single("derain_cosine")` — 单实验诊断
- `compare_experiments(["derain_cosine", "derain_linear", "derain_constant"])` — theta 对比
