"""The ONE metrics function every tier calls. Keeps comparison apples-to-apples.
Reports positive(anomaly)-class Precision, Recall, F1, and False Positive Rate.
"""
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def evaluate(y_true, y_pred, tier, feature, split):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "tier": tier, "feature": feature, "split": split,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "fpr":       round(fpr, 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
