import matplotlib
matplotlib.use("Agg")  # headless: no display available, only saving PNGs
import matplotlib.pyplot as plt
import numpy as np


def save_block_rate_chart(block_rate_df, path):
    categories = block_rate_df.index.tolist()
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, block_rate_df["Phase 1"], width, label="Phase 1 (Layer 1 only)")
    ax.bar(x + width / 2, block_rate_df["Phase 2"], width, label="Phase 2 (Layer 1 + Layer 2)")
    ax.set_ylabel("Block Rate (%)")
    ax.set_title("Attack Block Rate by Category: Phase 1 vs Phase 2")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.bar_label(ax.containers[0], fmt="%.0f%%", padding=2, fontsize=8)
    ax.bar_label(ax.containers[1], fmt="%.0f%%", padding=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_fpr_chart(overall_metrics_df, path):
    fig, ax = plt.subplots(figsize=(5, 5.5))
    fpr_values = overall_metrics_df["False Positive Rate (%)"].tolist()
    bars = ax.bar(overall_metrics_df["Phase"], fpr_values, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("False Positive Rate (%)")
    ax.set_title("False Positive Rate: Phase 1 vs Phase 2")
    ax.set_ylim(0, max(10, max(fpr_values) * 1.5 if max(fpr_values) > 0 else 10))
    ax.bar_label(bars, fmt="%.1f%%", padding=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
