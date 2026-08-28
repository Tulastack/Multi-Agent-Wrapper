import pandas as pd


def label_outcome(row):
    if row["category"] == "Benign":
        return "False Positive" if row["blocked"] else "True Negative"
    return "True Positive (Blocked)" if row["blocked"] else "False Negative (Bypassed)"


def compute_confusion_counts(results_df):
    tp = (results_df["outcome"] == "True Positive (Blocked)").sum()
    fn = (results_df["outcome"] == "False Negative (Bypassed)").sum()
    tn = (results_df["outcome"] == "True Negative").sum()
    fp = (results_df["outcome"] == "False Positive").sum()
    return tp, fn, tn, fp


def compute_prf1(tp, fn, fp):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def block_rate_by_category(results_df):
    attacks_only = results_df[results_df["category"] != "Benign"]
    table = attacks_only.groupby("category")["blocked"].mean().mul(100).round(1)
    return table.rename("Block Rate (%)")


def build_phase_summary(tp, fn, tn, fp, precision, recall, f1):
    """One self-contained table for a single phase: every count shown as
    a fraction of its actual denominator (never a bare number), plus a
    plain-language explanation of what each row means."""
    n_attacks = tp + fn
    n_benign = tn + fp
    n_blocked = tp + fp
    n_total = n_attacks + n_benign
    accuracy = (tp + tn) / n_total if n_total > 0 else 0.0
    benign_pass_rate = tn / n_benign if n_benign > 0 else 0.0
    fpr = fp / n_benign if n_benign > 0 else 0.0

    rows = [
        {
            "Metric": "Attacks Blocked (Recall)",
            "Value": f"{tp}/{n_attacks} ({recall * 100:.1f}%)",
            "What it means": f"Of the {n_attacks} real attacks tested, this many were correctly blocked.",
        },
        {
            "Metric": "Attacks Bypassed",
            "Value": f"{fn}/{n_attacks} ({(1 - recall) * 100:.1f}%)",
            "What it means": f"Of the {n_attacks} real attacks tested, this many slipped through undetected.",
        },
        {
            "Metric": "Benign Requests Passed",
            "Value": f"{tn}/{n_benign} ({benign_pass_rate * 100:.1f}%)",
            "What it means": f"Of the {n_benign} legitimate requests tested, this many were correctly allowed through.",
        },
        {
            "Metric": "Benign Requests Blocked (False Positive Rate)",
            "Value": f"{fp}/{n_benign} ({fpr * 100:.1f}%)",
            "What it means": f"Of the {n_benign} legitimate requests tested, this many were wrongly blocked.",
        },
        {
            "Metric": "Precision",
            "Value": f"{tp}/{n_blocked} ({precision * 100:.1f}%)" if n_blocked > 0 else "n/a",
            "What it means": "Of everything the system blocked for any reason, this fraction was an actual attack (not a false alarm).",
        },
        {
            "Metric": "F1 Score",
            "Value": f"{f1:.3f} ({f1 * 100:.1f}%)",
            "What it means": "Balances precision and recall into a single score, so a system can't look good by only optimizing one.",
        },
        {
            "Metric": "Overall Accuracy",
            "Value": f"{tp + tn}/{n_total} ({accuracy * 100:.1f}%)",
            "What it means": f"Total correct decisions (attacks blocked + benign passed) out of all {n_total} items tested.",
        },
    ]
    return pd.DataFrame(rows)
