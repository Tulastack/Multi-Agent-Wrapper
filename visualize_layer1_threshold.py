import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

from calibration import LAYER1_THRESHOLD
from dataset import all_items
from wrapper import attack_reference_phrases, run_layer1_batch

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
reference_vectors = embedding_model.encode(attack_reference_phrases, show_progress_bar=False)

texts = [item["text"] for item in all_items]
categories = [item["category"] for item in all_items]
blocked, scores, _ = run_layer1_batch(texts, embedding_model, reference_vectors, LAYER1_THRESHOLD)

category_order = ["Benign", "Direct Overrides", "Obfuscation", "Role-Play", "Data Exfiltration", "Agent Manipulation"]
colors = {"Benign": "#4C72B0"}
for cat in category_order[1:]:
    colors[cat] = "#DD8452"

benign_scores = [s for s, c in zip(scores, categories) if c == "Benign"]
highest_benign = max(benign_scores)
lowest_direct_override = min(s for s, c in zip(scores, categories) if c == "Direct Overrides")

# The other 3 attack categories dip below the threshold (and some below
# the benign cluster entirely) -- that's the honest part of the story.
below_threshold_other_cats = [
    (s, c) for s, c in zip(scores, categories)
    if c not in ("Benign", "Direct Overrides") and s < LAYER1_THRESHOLD
]

fig, ax = plt.subplots(figsize=(9, 6))
rng = np.random.default_rng(0)

for row, cat in enumerate(category_order):
    cat_scores = [s for s, c in zip(scores, categories) if c == cat]
    jitter = rng.uniform(-0.15, 0.15, size=len(cat_scores))
    ax.scatter(cat_scores, [row + j for j in jitter], color=colors[cat], alpha=0.8, s=45,
               edgecolors="white", linewidths=0.5, zorder=3)

ax.axvline(LAYER1_THRESHOLD, color="black", linestyle="--", linewidth=1.5, zorder=2)
ax.text(LAYER1_THRESHOLD + 0.01, len(category_order) - 0.4, f"threshold = {LAYER1_THRESHOLD}",
        rotation=90, va="top", fontsize=9)

# The REAL clean gap the threshold was calibrated on: Benign vs. Direct
# Overrides only (the category with explicit "jailbreak" vocabulary).
ax.axvspan(highest_benign, lowest_direct_override, color="green", alpha=0.12, zorder=1)
ax.text((highest_benign + lowest_direct_override) / 2, -1.15,
        f"real gap the threshold was calibrated on:\nBenign vs. Direct Overrides only ({highest_benign:.3f} to {lowest_direct_override:.3f})",
        ha="center", fontsize=8, color="#2c6e3f")

# Call out that Obfuscation/Role-Play/Data Exfiltration/Agent Manipulation
# items fall below that same threshold -- no vocabulary to match against.
ax.annotate(
    f"{len(below_threshold_other_cats)} of {sum(1 for c in categories if c not in ('Benign', 'Direct Overrides'))} attacks in the\n"
    "other 4 categories score BELOW\nthe threshold -- no attack\nvocabulary for Layer 1 to match",
    xy=(0.08, 4.5), xytext=(0.5, 4.6), fontsize=8, color="#a83232",
    arrowprops=dict(arrowstyle="->", color="#a83232"),
)

ax.set_yticks(range(len(category_order)))
ax.set_yticklabels(category_order)
ax.invert_yaxis()
ax.set_xlabel("Layer 1 score (max similarity to a reference attack phrase)")
ax.set_title("Layer 1 scores by category, with the calibrated threshold (0.32)", pad=14)
ax.set_ylim(len(category_order) - 0.5, -1.6)

legend_handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["Benign"], markersize=8, label="Benign"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["Direct Overrides"], markersize=8, label="Attack"),
]
ax.legend(handles=legend_handles, loc="upper right")

fig.tight_layout()
fig.savefig("output/layer1_threshold.png", dpi=150)
plt.close(fig)
print("Saved: output/layer1_threshold.png")
print(f"Highest benign score: {highest_benign:.3f}")
print(f"Lowest Direct Overrides score: {lowest_direct_override:.3f}")
print(f"Attacks in other 4 categories scoring below threshold: {len(below_threshold_other_cats)}")
