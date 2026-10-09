"""Step 3+4 — group parsed lines into per-block sessions, attach labels,
freeze a chronological 70/10/20 split (and a random split for the ablation).
Output: data/processed/sessions.parquet, data/processed/split_index.json
"""
import json
import numpy as np
import pandas as pd
from config import INTERIM, RAW, PROCESSED, SPLIT, SEED

def sessionize():
    df = pd.read_parquet(INTERIM / "structured_logs.parquet")
    # a line can reference several blocks -> explode so each (line,block) is a row
    df = df.assign(block_id=df["block_ids"].str.split(",")).explode("block_id")
    df = df[df["block_id"].astype(bool)]
    df = df.sort_values("LineId")  # LineId is time order

    grp = df.groupby("block_id")
    sessions = pd.DataFrame({
        "event_sequence": grp["EventId"].apply(list),
        "first_line": grp["LineId"].min(),
    }).reset_index()

    labels = pd.read_csv(RAW / "anomaly_label.csv")
    labels.columns = [c.strip() for c in labels.columns]
    lab_col = "Label" if "Label" in labels.columns else labels.columns[-1]
    blk_col = "BlockId" if "BlockId" in labels.columns else labels.columns[0]
    labels = labels.rename(columns={blk_col: "block_id", lab_col: "Label"})
    labels["y"] = (labels["Label"].str.strip().str.lower() == "anomaly").astype(int)
    sessions = sessions.merge(labels[["block_id", "y"]], on="block_id", how="inner")

    # chronological split by first-event time
    sessions = sessions.sort_values("first_line").reset_index(drop=True)
    n = len(sessions)
    a, b = int(n * SPLIT[0]), int(n * (SPLIT[0] + SPLIT[1]))
    chrono = {"train": sessions["block_id"].iloc[:a].tolist(),
              "val":   sessions["block_id"].iloc[a:b].tolist(),
              "test":  sessions["block_id"].iloc[b:].tolist()}

    # random split (ablation)
    rng = np.random.default_rng(SEED)
    perm = sessions["block_id"].sample(frac=1.0, random_state=SEED).tolist()
    rnd = {"train": perm[:a], "val": perm[a:b], "test": perm[b:]}

    sessions.to_parquet(PROCESSED / "sessions.parquet", index=False)
    (PROCESSED / "split_index.json").write_text(
        json.dumps({"chronological": chrono, "random": rnd}))
    return sessions, chrono

if __name__ == "__main__":
    s, chrono = sessionize()
    print(f"[sessionize] {len(s)} sessions, anomaly rate {100*s['y'].mean():.2f}%")
    print(f"  train/val/test = {len(chrono['train'])}/{len(chrono['val'])}/{len(chrono['test'])}")
    print(f"  median session length = {int(s['event_sequence'].apply(len).median())}")
