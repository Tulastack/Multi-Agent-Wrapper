import time

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.linear_model import LogisticRegression

from run_experiment import LAYER1_THRESHOLD, all_items, benign_baseline_texts
from wrapper import attack_reference_phrases, run_layer1_batch

N_RUNS = 5


# The old sequential Layer 1 -> conditional NLI Layer 2 pipeline, kept
# self-contained here (removed from wrapper.py, which now only contains
# the shipped design) purely so this benchmark still has something to
# compare against.
def get_label_index(cross_encoder, label_name):
    id2label = cross_encoder.model.config.id2label
    for idx, name in id2label.items():
        if name.lower() == label_name.lower():
            return int(idx)
    raise ValueError(f"No '{label_name}' label found in model labels: {id2label}")


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def run_old_two_layer_pipeline(texts, embedding_model, reference_vectors, cross_encoder,
                                contradiction_idx, baseline_texts, layer1_threshold, layer2_threshold):
    blocked1, _, _ = run_layer1_batch(texts, embedding_model, reference_vectors, layer1_threshold)
    texts_for_layer2 = [t for t, blocked in zip(texts, blocked1) if not blocked]
    if texts_for_layer2:
        pairs = [(baseline, text) for text in texts_for_layer2 for baseline in baseline_texts]
        probs = softmax(cross_encoder.predict(pairs, show_progress_bar=False), axis=1)
        probs[:, contradiction_idx].reshape(len(texts_for_layer2), len(baseline_texts)).max(axis=1)
    return blocked1


print("Loading models...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/nli-deberta-v3-small")
contradiction_idx = get_label_index(cross_encoder, "contradiction")

reference_vectors = embedding_model.encode(attack_reference_phrases, show_progress_bar=False)

all_texts = [item["text"] for item in all_items]
all_categories = [item["category"] for item in all_items]
labels = np.array([0 if c == "Benign" else 1 for c in all_categories])

# One-time setup cost: fitting LogReg happens ONCE, offline, same as how
# Layer 1's reference_vectors are computed once and reused. Not counted
# in per-request latency below.
print("Fitting logistic regression classifier (one-time setup)...")
X_train = embedding_model.encode(all_texts, show_progress_bar=False)
clf = LogisticRegression(max_iter=2000, C=30.0)
clf.fit(X_train, labels)

LAYER2_THRESHOLD_NLI = 0.501  # calibrated earlier for the NLI cross-encoder


def summarize(name, times, n_items):
    avg = sum(times) / len(times)
    print(f"{name}: {avg:.4f}s total for {n_items} items, {avg / n_items * 1000:.3f} ms/item")
    return avg


print(f"\nBenchmarking each pipeline over {N_RUNS} runs on all {len(all_texts)} items...\n")

# ---------------------------------------------------------------------
# OLD: Layer 1 (embed + dot product) -> conditional NLI cross-encoder
# on items Layer 1 didn't catch (the pipeline we already shipped).
# ---------------------------------------------------------------------
old_times = []
for _ in range(N_RUNS):
    start = time.perf_counter()
    run_old_two_layer_pipeline(
        all_texts, embedding_model, reference_vectors, cross_encoder, contradiction_idx,
        benign_baseline_texts, LAYER1_THRESHOLD, LAYER2_THRESHOLD_NLI,
    )
    old_times.append(time.perf_counter() - start)

# ---------------------------------------------------------------------
# NEW: one embedding pass, Layer 1 (max similarity) AND logistic
# regression BOTH computed off that same embedding, combined with OR.
# No cross-encoder, no conditional branching.
# ---------------------------------------------------------------------
def run_parallel_pipeline(texts):
    vecs = embedding_model.encode(texts, show_progress_bar=False)
    layer1_scores = np.dot(vecs, reference_vectors.T).max(axis=1)
    layer1_blocked = layer1_scores > LAYER1_THRESHOLD
    logreg_blocked = clf.predict(vecs).astype(bool)
    return layer1_blocked | logreg_blocked


new_times = []
for _ in range(N_RUNS):
    start = time.perf_counter()
    run_parallel_pipeline(all_texts)
    new_times.append(time.perf_counter() - start)

# ---------------------------------------------------------------------
# For reference: the embedding step alone, to show where the new
# pipeline's time is actually going (it's basically all here).
# ---------------------------------------------------------------------
embed_times = []
for _ in range(N_RUNS):
    start = time.perf_counter()
    embedding_model.encode(all_texts, show_progress_bar=False)
    embed_times.append(time.perf_counter() - start)

n = len(all_texts)
avg_old = summarize("OLD  (Layer 1 -> conditional NLI cross-encoder)", old_times, n)
avg_new = summarize("NEW  (Layer 1 + LogReg, parallel, shared embedding)", new_times, n)
avg_embed = summarize("     embedding step alone (both pipelines pay this)", embed_times, n)

print(f"\nSpeedup: {avg_old / avg_new:.1f}x faster")
print(f"The NEW pipeline's cost is ~{avg_new / avg_embed:.1f}x the bare embedding cost "
      f"— i.e. almost all of its time IS the embedding step, classification is nearly free.")
