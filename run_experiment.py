import os
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from wrapper import attack_reference_phrases, run_layer1_batch, run_parallel_pipeline
from dataset import all_items
from calibration import LAYER1_THRESHOLD, LOGREG_C
from analysis.metrics import label_outcome, compute_confusion_counts, compute_prf1, block_rate_by_category
from analysis.charts import save_block_rate_chart

OUTPUT_DIR = "output"
N_TIMING_RUNS = 5


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("STEP 0: Loading models")
    print("=" * 70)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    reference_vectors = embedding_model.encode(attack_reference_phrases, show_progress_bar=False)

    all_texts = [item["text"] for item in all_items]
    all_categories = [item["category"] for item in all_items]
    labels = np.array([0 if c == "Benign" else 1 for c in all_categories])  # 1 = attack
    embeddings = embedding_model.encode(all_texts, show_progress_bar=False)

    # This is the classifier that would actually ship: fit on every
    # labeled example we have. It's used below for the latency
    # benchmark. It is NOT used to report Phase 2's accuracy — see the
    # leave-one-out note in Phase 2.
    classifier = LogisticRegression(max_iter=2000, C=LOGREG_C)
    classifier.fit(embeddings, labels)
    print(f"Layer 1: rule-based similarity to {len(attack_reference_phrases)} reference attack phrases.")
    print(f"Layer 2: logistic regression (C={LOGREG_C}), fit on all {len(all_texts)} labeled examples.")

    # -----------------------------------------------------------------
    # PHASE 1 — Layer 1 only
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1: PHASE 1 — Layer 1 only, 50 attacks + 20 benign")
    print("=" * 70)

    blocked1, score1, pattern1 = run_layer1_batch(all_texts, embedding_model, reference_vectors, LAYER1_THRESHOLD)
    phase1_df = pd.DataFrame({
        "text": all_texts,
        "category": all_categories,
        "phase": "Phase 1",
        "blocked": blocked1,
        "layer1_score": score1,
        "matched_pattern": pattern1,
    })
    phase1_df["outcome"] = phase1_df.apply(label_outcome, axis=1)
    print(f"Phase 1 complete: {phase1_df['blocked'].sum()} of {len(phase1_df)} items blocked.")

    # -----------------------------------------------------------------
    # PHASE 2 — Layer 1 + Layer 2, run in PARALLEL (not a cascade — see
    # wrapper/ for why). Layer 2's block decision here comes from
    # LEAVE-ONE-OUT cross-validation: each item is predicted by a
    # classifier that never saw it while fitting. This is the honest
    # measure of how Layer 2 generalizes; evaluating the classifier above
    # (fit on all 70) on those same 70 examples would only show what it
    # memorized, not what it learned.
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: PHASE 2 — Layer 1 + Layer 2 (parallel), leave-one-out evaluation")
    print("=" * 70)

    loo_probabilities = cross_val_predict(
        LogisticRegression(max_iter=2000, C=LOGREG_C), embeddings, labels,
        cv=LeaveOneOut(), method="predict_proba",
    )[:, 1]
    layer2_blocked = loo_probabilities > 0.5

    final_blocked = blocked1 | layer2_blocked
    decided_by = np.full(len(all_texts), "Passed", dtype=object)
    decided_by[layer2_blocked] = "Layer 2"
    decided_by[blocked1 & ~layer2_blocked] = "Layer 1"
    decided_by[blocked1 & layer2_blocked] = "Both"

    phase2_df = pd.DataFrame({
        "text": all_texts,
        "category": all_categories,
        "phase": "Phase 2",
        "blocked": final_blocked,
        "layer1_score": score1,
        "layer2_score": loo_probabilities,
        "decided_by": list(decided_by),
    })
    phase2_df["outcome"] = phase2_df.apply(label_outcome, axis=1)
    print(f"Phase 2 complete: {phase2_df['blocked'].sum()} of {len(phase2_df)} items blocked.")
    print("(Layer 2 numbers are leave-one-out cross-validated, the honest estimate. "
          "The classifier actually shipped is refit on all 70 examples, used below "
          "for the latency benchmark.)")

    # -----------------------------------------------------------------
    # STEP 3 — Latency benchmark (batched, 5 runs averaged). Uses the
    # shipped classifier fit above — fitting is one-time setup, not
    # part of per-request latency.
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"STEP 3: Latency benchmark ({N_TIMING_RUNS} runs, Layer 1 + Layer 2 in parallel)")
    print("=" * 70)

    run_times = []
    for run_i in range(N_TIMING_RUNS):
        start = time.perf_counter()
        run_parallel_pipeline(all_texts, embedding_model, reference_vectors, classifier, LAYER1_THRESHOLD)
        elapsed = time.perf_counter() - start
        run_times.append(elapsed)
        print(f"  Run {run_i + 1}/{N_TIMING_RUNS}: {elapsed:.3f}s total, "
              f"{elapsed / len(all_texts) * 1000:.2f} ms/item")

    avg_total = sum(run_times) / len(run_times)
    avg_per_item_ms = avg_total / len(all_texts) * 1000
    print(f"\nAverage across {N_TIMING_RUNS} runs: {avg_total:.3f}s total, "
          f"{avg_per_item_ms:.2f} ms/item for {len(all_texts)} items "
          f"(one shared embedding pass; both layers scored off it).")

    # -----------------------------------------------------------------
    # STEP 4 — Metrics: per-phase precision/recall/F1, confusion counts,
    # block rate by category, false positive rate.
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Metrics")
    print("=" * 70)

    phase_metrics_rows = []
    block_rate_tables = {}
    for phase_name, df in [("Phase 1", phase1_df), ("Phase 2", phase2_df)]:
        tp, fn, tn, fp = compute_confusion_counts(df)
        precision, recall, f1 = compute_prf1(tp, fn, fp)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        phase_metrics_rows.append({
            "Phase": phase_name,
            "True Positives (Attacks Blocked)": tp,
            "False Negatives (Attacks Bypassed)": fn,
            "True Negatives (Benign Passed)": tn,
            "False Positives (Benign Blocked)": fp,
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1 Score": round(f1, 3),
            "False Positive Rate (%)": round(fpr * 100, 1),
        })
        block_rate_tables[phase_name] = block_rate_by_category(df)

    overall_metrics_df = pd.DataFrame(phase_metrics_rows)

    block_rate_df = pd.DataFrame(block_rate_tables)
    block_rate_df.index.name = "Category"

    print("\n--- Block Rate by Category (%) ---")
    print(block_rate_df.to_string())

    print("\n--- Confusion Matrix + Precision/Recall/F1 by Phase ---")
    print(overall_metrics_df.to_string(index=False))

    # -----------------------------------------------------------------
    # STEP 5 — Save CSVs
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: Saving tables (CSV) and charts (PNG)")
    print("=" * 70)

    block_rate_csv = f"{OUTPUT_DIR}/block_rate_by_category.csv"
    metrics_csv = f"{OUTPUT_DIR}/phase_metrics.csv"
    block_rate_df.to_csv(block_rate_csv)
    overall_metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved: {block_rate_csv}")
    print(f"Saved: {metrics_csv}")

    # -----------------------------------------------------------------
    # STEP 6 — Charts
    # -----------------------------------------------------------------
    block_rate_png = f"{OUTPUT_DIR}/block_rate_by_category.png"
    save_block_rate_chart(block_rate_df, block_rate_png)
    print(f"Saved: {block_rate_png}")

    print("\nDone.")


if __name__ == "__main__":
    main()
