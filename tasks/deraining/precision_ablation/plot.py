"""
Plot precision ablation results: loss, PSNR, and training time comparison.

Usage:
    python tasks/deraining/precision_ablation/plot.py
"""
import json
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

PRECISIONS = ["fp32", "fp16", "bf16"]
OUTPUT_ROOT = "outputs"
EXP_PREFIX = "precision_"


def load_jsonl(path: str) -> list[dict]:
    records = []
    if not os.path.exists(path):
        print(f"  [WARN] missing: {path}")
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    # Load data
    train_data = {}
    val_data = {}
    for prec in PRECISIONS:
        exp_dir = os.path.join(OUTPUT_ROOT, f"{EXP_PREFIX}{prec}", "log")
        train_data[prec] = load_jsonl(os.path.join(exp_dir, "train_metrics.jsonl"))
        val_data[prec] = load_jsonl(os.path.join(exp_dir, "val_metrics.jsonl"))

    # Check data availability
    available = [p for p in PRECISIONS if train_data[p] and val_data[p]]
    if not available:
        print("No results found. Run the ablation scripts first.")
        return
    print(f"Found data for: {', '.join(available)}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Train Loss ---
    ax = axes[0][0]
    for prec in available:
        iterations = [r["iter"] for r in train_data[prec]]
        losses = [r["loss"] for r in train_data[prec]]
        ax.plot(iterations, losses, alpha=0.6, linewidth=0.6, label=prec.upper())
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Val PSNR ---
    ax = axes[0][1]
    for prec in available:
        iterations = [r["iter"] for r in val_data[prec]]
        psnrs = [r["psnr"] for r in val_data[prec]]
        ax.plot(iterations, psnrs, marker="o", markersize=3, label=prec.upper())
    ax.set_xlabel("Iteration")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Validation PSNR")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Total Time ---
    ax = axes[1][0]
    times = []
    labels = []
    for prec in available:
        total_h = None
        for r in reversed(train_data[prec]):
            if "total_time_hours" in r:
                total_h = r["total_time_hours"]
                break
        if total_h is not None:
            times.append(total_h)
            labels.append(prec.upper())
    if times:
        bars = ax.bar(labels, times, color=["#2ecc71", "#3498db", "#9b59b6"][:len(labels)])
        ax.bar_label(bars, fmt="%.1fh")
    ax.set_ylabel("Hours")
    ax.set_title("Total Training Time")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Step Time ---
    ax = axes[1][1]
    for prec in available:
        step_times = [r["step_time"] for r in train_data[prec]]
        if step_times:
            ax.boxplot(
                [step_times],
                positions=[available.index(prec)],
                labels=[prec.upper()],
                widths=0.5,
                showfliers=False,
            )
    ax.set_ylabel("Step Time (s)")
    ax.set_title("Step Time Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Precision Ablation: FP32 vs FP16 vs BF16", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_dir = os.path.join(OUTPUT_ROOT, "precision_ablation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "precision_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
