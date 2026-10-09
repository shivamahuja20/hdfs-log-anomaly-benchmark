"""ONE-TIME builder for the classroom demo's tier-by-tier comparison.
Produces results/predictions.csv with EVERY tier's verdict for each test session,
so demo.py can show all four tiers side-by-side, instantly, in class.

Run once (retrains the DeepLog LSTM, a few minutes):
    python src/make_demo_data.py
"""
import re, json, pickle
import numpy as np
import pandas as pd
from config import PROCESSED, INTERIM, RESULTS, MODELS
from features import count_matrix
import tier2_deeplog

KW = re.compile(r"error|exception|failed|fatal|timed out|received exception", re.I)

def main():
    sessions = pd.read_parquet(PROCESSED / "sessions.parquet").set_index("block_id")
    split = json.loads((PROCESSED / "split_index.json").read_text())["chronological"]
    test = split["test"]
    tpl = pd.read_csv(INTERIM / "templates.csv")
    bad_events = set(tpl.loc[tpl["template"].str.contains(KW), "EventId"])
    with (PROCESSED / "features.pkl").open("rb") as f:
        vocab = pickle.load(f)["vocab"]

    df = pd.DataFrame(index=test)
    df["y_true"] = [int(sessions.loc[b, "y"]) for b in test]

    # ---- Tier 0: keyword rule ----
    df["tier0"] = [int(any(e in bad_events for e in sessions.loc[b, "event_sequence"]))
                   for b in test]

    # ---- Tier 1: reload PCA + Isolation Forest ----
    with (MODELS / "tier1.pkl").open("rb") as f:
        m = pickle.load(f)
    Xte = count_matrix(sessions.loc[test, "event_sequence"].tolist(), vocab)
    Zte = np.log1p(Xte)
    recon = m["pca"].inverse_transform(m["pca"].transform(Zte))
    q = np.sum((Zte - recon) ** 2, axis=1)
    df["tier1_pca"] = (q > m["t_pca"]).astype(int)
    df["tier1_iforest"] = (m["iso"].predict(Zte) == -1).astype(int)

    # ---- Tier 2: DeepLog (retrains + exports pred_tier2_chronological.csv) ----
    print("  running Tier 2 (this retrains the model, a few minutes)...")
    tier2_deeplog.run("chronological")
    t2 = pd.read_csv(RESULTS / "pred_tier2_chronological.csv").set_index("block_id")
    df["tier2"] = t2["tier2"].reindex(df.index).astype("Int64")

    # ---- Tier 3: LLM (from cached explanations; only the sampled sessions) ----
    expl = json.loads((RESULTS / "llm_explanations.json").read_text())
    lab = {e["block_id"]: (1 if str(e.get("label","")).lower().startswith("anom") else 0)
           for e in expl}
    rsn = {e["block_id"]: (e.get("reason") or "") for e in expl}
    df["tier3"] = [lab.get(b, pd.NA) for b in test]
    df["tier3_reason"] = [rsn.get(b, "") for b in test]

    df.index.name = "block_id"
    df.to_csv(RESULTS / "predictions.csv")
    n_all4 = df["tier3"].notna().sum()
    print(f"  wrote results/predictions.csv  ({len(df)} sessions; "
          f"{n_all4} have all four tiers incl. LLM)")

if __name__ == "__main__":
    main()
