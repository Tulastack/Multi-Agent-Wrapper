"""Diagnostic: would a logistic regression classifier (trained on Layer 1
embeddings) do better than NLI at catching Data Exfiltration and Agent
Manipulation attacks?

Compares two evaluation methods on the SAME model to make a point
concrete: with 70 examples and 384-dim embeddings, a linear classifier
can trivially memorize the training set. The real question is whether it
generalizes, which resubstitution (train==test) cannot tell you.

1. RESUBSTITUTION: fit on all 70, evaluate on all 70 (what "fit and
   check the same data" looks like — this is what we did for both
   layers' thresholds so far).
2. LEAVE-ONE-OUT CROSS-VALIDATION: for each of the 70 items, fit on the
   other 69 and predict the held-out one. This is the honest estimate of
   how it'd perform on an attack it hasn't seen before.

Not part of the paper's pipeline — throwaway investigation code.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from run_experiment import all_items

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [item["text"] for item in all_items]
categories = [item["category"] for item in all_items]
labels = np.array([0 if c == "Benign" else 1 for c in categories])  # 1 = attack

print("Encoding all 70 items...")
X = embedding_model.encode(texts, show_progress_bar=False)

clf = LogisticRegression(max_iter=2000, C=1.0)

# ---------------------------------------------------------------------
# 1. Resubstitution: fit and evaluate on the same 70 examples.
# ---------------------------------------------------------------------
clf.fit(X, labels)
resub_preds = clf.predict(X)

# ---------------------------------------------------------------------
# 2. Leave-one-out cross-validation: honest, held-out predictions.
# ---------------------------------------------------------------------
print("Running leave-one-out cross-validation (70 fits)...")
loo_preds = cross_val_predict(clf, X, labels, cv=LeaveOneOut())


def report(name, preds):
    df = pd.DataFrame({"category": categories, "label": labels, "pred": preds})
    print(f"\n--- {name} ---")
    rows = []
    for cat in ["Direct Overrides", "Obfuscation", "Role-Play", "Data Exfiltration", "Agent Manipulation"]:
        sub = df[df["category"] == cat]
        recall = (sub["pred"] == 1).mean()
        rows.append({"Category": cat, "Caught (%)": round(recall * 100, 1)})
    benign = df[df["category"] == "Benign"]
    fpr = (benign["pred"] == 1).mean()
    rows.append({"Category": "Benign (this is FPR, not a catch rate)", "Caught (%)": round(fpr * 100, 1)})
    print(pd.DataFrame(rows).to_string(index=False))
    return fpr


report("RESUBSTITUTION (fit and test on the same 70 examples)", resub_preds)
fpr_loo = report("LEAVE-ONE-OUT CV (honest: predict each item using a model that never saw it)", loo_preds)

print("\n" + "=" * 70)
print("The gap between these two tables IS the overfitting.")
print("Resubstitution tells you what the model memorized.")
print("Leave-one-out tells you what it actually learned.")
print("=" * 70)

# ---------------------------------------------------------------------
# 3. Regularization sweep: does a stronger (or weaker) penalty trade off
#    recall on Data Exfiltration/Agent Manipulation against FPR?
#    All numbers below are leave-one-out (honest), never resubstitution.
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("REGULARIZATION SWEEP (all leave-one-out, honest numbers)")
print("=" * 70)
sweep_rows = []
for C in [0.001, 0.01, 0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
    sweep_clf = LogisticRegression(max_iter=2000, C=C)
    sweep_preds = cross_val_predict(sweep_clf, X, labels, cv=LeaveOneOut())
    df = pd.DataFrame({"category": categories, "label": labels, "pred": sweep_preds})
    benign_fpr = (df[df["category"] == "Benign"]["pred"] == 1).mean()
    data_exfil_recall = (df[df["category"] == "Data Exfiltration"]["pred"] == 1).mean()
    agent_manip_recall = (df[df["category"] == "Agent Manipulation"]["pred"] == 1).mean()
    overall_recall = (df[df["label"] == 1]["pred"] == 1).mean()
    sweep_rows.append({
        "C": C,
        "FPR (%)": round(benign_fpr * 100, 1),
        "Data Exfil Recall (%)": round(data_exfil_recall * 100, 1),
        "Agent Manip Recall (%)": round(agent_manip_recall * 100, 1),
        "Overall Recall (%)": round(overall_recall * 100, 1),
    })
print(pd.DataFrame(sweep_rows).to_string(index=False))
