"""Tier 1 — classical ML on event-count vectors, trained on NORMAL sessions only.
PCA reconstruction-error (Q-statistic) + Isolation Forest. Thresholds picked on val.
"""
import json, pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from config import PROCESSED, MODELS, PCA_VAR, IFOREST_ESTIMATORS, IFOREST_CONTAM, SEED
from features import count_matrix
from evaluate import evaluate

def _mats(split_name):
    with (PROCESSED / "features.pkl").open("rb") as f:
        F = pickle.load(f)
    s = pd.read_parquet(PROCESSED / "sessions.parquet").set_index("block_id")
    vocab = F["vocab"]
    split = json.loads((PROCESSED / "split_index.json").read_text())[split_name]
    def mat(ids):
        seqs = s.loc[ids, "event_sequence"].tolist()
        y = s.loc[ids, "y"].astype(int).to_numpy()
        return count_matrix(seqs, vocab), y
    return mat(split["train"]), mat(split["val"]), mat(split["test"])

def _pick_threshold(val_scores, y_val, train_normal_scores):
    """Maximise F1 on validation over a broad candidate grid; if val has too few
    anomalies to be informative, fall back to the 99th percentile of the
    train-normal score distribution (the classic novelty-detection rule)."""
    from sklearn.metrics import f1_score
    cand = np.unique(np.concatenate([
        np.quantile(train_normal_scores, np.linspace(0.80, 0.999, 40)),
        np.quantile(val_scores, np.linspace(0.50, 0.999, 40)),
    ]))
    best_t, best_f1 = None, -1.0
    for t in cand:
        f1 = f1_score(y_val, (val_scores > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    if best_f1 <= 0:                       # val uninformative -> percentile fallback
        best_t = np.quantile(train_normal_scores, 0.99)
    return best_t

def run(split_name="chronological"):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = _mats(split_name)
    norm = ytr == 0                       # train on normal only
    # log1p compresses count magnitudes but preserves anomaly-only columns
    # (z-scoring would zero out zero-variance normal columns and kill the signal)
    Ztr = np.log1p(Xtr[norm]); Zva = np.log1p(Xva); Zte = np.log1p(Xte)

    # --- PCA residual (Q-statistic) ---
    pca = PCA(n_components=PCA_VAR, svd_solver="full").fit(Ztr)
    def q_stat(Z):
        recon = pca.inverse_transform(pca.transform(Z))
        return np.sum((Z - recon) ** 2, axis=1)
    t_pca = _pick_threshold(q_stat(Zva), yva, q_stat(Ztr))
    pca_pred = (q_stat(Zte) > t_pca).astype(int)

    # --- Isolation Forest ---
    # Use the native contamination boundary (set to ~base rate). This is the
    # standard, honest operating point; a custom val-F1 threshold is unstable
    # here because iForest scores on 29-dim count vectors separate weakly.
    iso = IsolationForest(n_estimators=IFOREST_ESTIMATORS, contamination=IFOREST_CONTAM,
                          random_state=SEED).fit(Ztr)
    iso_pred = (iso.predict(Zte) == -1).astype(int)     # -1 = anomaly
    t_iso = None

    with (MODELS / "tier1.pkl").open("wb") as f:
        pickle.dump({"pca": pca, "iso": iso,
                     "t_pca": t_pca, "t_iso": t_iso}, f)
    return [evaluate(yte, pca_pred, "Tier1-PCA", "counts", split_name),
            evaluate(yte, iso_pred, "Tier1-iForest", "counts", split_name)]

if __name__ == "__main__":
    for r in run():
        print(r)
