"""Diagnostic: why is Layer 2 (NLI contradiction detection) so ineffective?

Investigates four things on the 44 items that actually reach Layer 2
(the ones Layer 1 doesn't catch):

1. Leave-one-out: the current design compares a benign item against ALL
   20 baselines, including itself. Does excluding self-pairs matter?
2. Aggregation strategy: max(contradiction) vs mean(contradiction) vs
   net signal (max contradiction - max entailment) vs comparing only
   against the single nearest baseline (by Layer 1 embedding similarity).
3. Pair order: NLI is directional (premise -> hypothesis is not the same
   as hypothesis -> premise). Does swapping the order change results?
4. For each strategy: does it actually separate benign from attack scores
   better than the current max-contradiction-over-all-20 approach?

Not part of the paper's pipeline — this is throwaway investigation code.
"""

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer

from wrapper import attack_reference_phrases, run_layer1_batch
from run_experiment import all_items, benign_baseline_texts, LAYER1_THRESHOLD


# Small NLI helpers, kept local here (removed from wrapper.py, which now
# only contains the shipped design) since this script exists specifically
# to investigate the abandoned NLI approach.
def get_label_index(cross_encoder, label_name):
    id2label = cross_encoder.model.config.id2label
    for idx, name in id2label.items():
        if name.lower() == label_name.lower():
            return int(idx)
    raise ValueError(f"No '{label_name}' label found in model labels: {id2label}")


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


print("Loading models...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/nli-deberta-v3-small")
contradiction_idx = get_label_index(cross_encoder, "contradiction")
entailment_idx = get_label_index(cross_encoder, "entailment")

reference_vectors = embedding_model.encode(attack_reference_phrases, show_progress_bar=False)

all_texts = [item["text"] for item in all_items]
all_categories = [item["category"] for item in all_items]

# Same subset Layer 2 actually sees in the real pipeline.
blocked1, _, _ = run_layer1_batch(all_texts, embedding_model, reference_vectors, LAYER1_THRESHOLD)
bypassed_mask = ~blocked1
item_texts = [t for t, b in zip(all_texts, bypassed_mask) if b]
item_categories = [c for c, b in zip(all_categories, bypassed_mask) if b]
is_benign = np.array([c == "Benign" for c in item_categories])
n_items = len(item_texts)
n_baselines = len(benign_baseline_texts)

print(f"Items reaching Layer 2: {n_items} ({is_benign.sum()} benign, {(~is_benign).sum()} attacks)\n")


def evaluate_scores(name, scores):
    """Given a deviation score per item, report separation quality and
    the confusion matrix at the empirical-midpoint threshold."""
    benign_scores = scores[is_benign]
    attack_scores = scores[~is_benign]
    highest_benign = benign_scores.max()
    lowest_attack = attack_scores.min()
    threshold = (highest_benign + lowest_attack) / 2
    blocked = scores > threshold

    tp = int(((~is_benign) & blocked).sum())
    fn = int(((~is_benign) & ~blocked).sum())
    tn = int((is_benign & ~blocked).sum())
    fp = int((is_benign & blocked).sum())
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    separated = highest_benign < lowest_attack
    print(f"--- {name} ---")
    print(f"  highest benign score: {highest_benign:.3f}   lowest attack score: {lowest_attack:.3f}   "
          f"{'CLEAN SEPARATION' if separated else 'OVERLAP'}")
    print(f"  threshold={threshold:.3f}  TP={tp} FN={fn} TN={tn} FP={fp}  "
          f"FPR={fpr:.1%}  precision={precision:.3f}  recall={recall:.3f}")
    print()
    return {"strategy": name, "fpr": fpr, "precision": precision, "recall": recall, "separated": separated}


# ---------------------------------------------------------------------
# 1. Reproduce the current design: max(contradiction) over ALL 20
#    baselines, self-pairs included.
# ---------------------------------------------------------------------
pairs = [(baseline, text) for text in item_texts for baseline in benign_baseline_texts]
raw = cross_encoder.predict(pairs, show_progress_bar=False)
probs = softmax(raw, axis=1)
contra = probs[:, contradiction_idx].reshape(n_items, n_baselines)
entail = probs[:, entailment_idx].reshape(n_items, n_baselines)

results = []
results.append(evaluate_scores("1. CURRENT: max(contradiction), self-pairs included", contra.max(axis=1)))

# ---------------------------------------------------------------------
# 2. Leave-one-out: mask out a benign item's comparison against itself.
# ---------------------------------------------------------------------
mask = np.ones((n_items, n_baselines), dtype=bool)
for i, text in enumerate(item_texts):
    for j, baseline in enumerate(benign_baseline_texts):
        if text == baseline:
            mask[i, j] = False

contra_loo = np.where(mask, contra, np.nan)
entail_loo = np.where(mask, entail, np.nan)
results.append(evaluate_scores("2. max(contradiction), leave-one-out (exclude self-pair)", np.nanmax(contra_loo, axis=1)))

# ---------------------------------------------------------------------
# 3. Mean instead of max (dampens one spurious high pair).
# ---------------------------------------------------------------------
results.append(evaluate_scores("3. mean(contradiction), leave-one-out", np.nanmean(contra_loo, axis=1)))

# ---------------------------------------------------------------------
# 4. Net signal: max(contradiction) - max(entailment).
# ---------------------------------------------------------------------
net_signal = np.nanmax(contra_loo, axis=1) - np.nanmax(entail_loo, axis=1)
results.append(evaluate_scores("4. max(contradiction) - max(entailment), leave-one-out", net_signal))

# ---------------------------------------------------------------------
# 5. Compare only against the single most-similar baseline (via Layer 1
#    embedding cosine similarity), instead of all 20.
# ---------------------------------------------------------------------
item_vecs = embedding_model.encode(item_texts, show_progress_bar=False)
baseline_vecs = embedding_model.encode(benign_baseline_texts, show_progress_bar=False)
item_norm = item_vecs / np.linalg.norm(item_vecs, axis=1, keepdims=True)
baseline_norm = baseline_vecs / np.linalg.norm(baseline_vecs, axis=1, keepdims=True)
cos_sim = np.dot(item_norm, baseline_norm.T)
cos_sim_masked = np.where(mask, cos_sim, -np.inf)
nearest_idx = np.argmax(cos_sim_masked, axis=1)
nearest_contra = contra[np.arange(n_items), nearest_idx]
results.append(evaluate_scores("5. contradiction vs single NEAREST baseline only (by embedding similarity)", nearest_contra))

# ---------------------------------------------------------------------
# 6. Pair-order symmetry check: does swapping premise/hypothesis change
#    the contradiction score? (NLI is directional.)
# ---------------------------------------------------------------------
print("--- 6. Pair-order symmetry check (5 example items) ---")
sample_idx = [0, 1, 10, 20, 30] if n_items >= 31 else list(range(min(5, n_items)))
for i in sample_idx:
    text = item_texts[i]
    cat = item_categories[i]
    j = nearest_idx[i]
    baseline = benign_baseline_texts[j]
    forward = cross_encoder.predict([(baseline, text)], show_progress_bar=False)
    backward = cross_encoder.predict([(text, baseline)], show_progress_bar=False)
    fp_probs = softmax(forward, axis=1)[0]
    bp_probs = softmax(backward, axis=1)[0]
    print(f"  [{cat}] \"{text[:60]}...\"")
    print(f"    vs baseline \"{baseline[:60]}...\"")
    print(f"    (baseline->item) contradiction={fp_probs[contradiction_idx]:.3f}  "
          f"(item->baseline) contradiction={bp_probs[contradiction_idx]:.3f}")
print()

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
summary_df = pd.DataFrame(results)
print("=" * 70)
print("SUMMARY (calibrated at each strategy's own empirical-midpoint threshold)")
print("=" * 70)
print(summary_df.to_string(index=False))
