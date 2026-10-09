"""Build sessions directly from the OFFICIAL LogHub HDFS_v1 preprocessed files
(Event_traces.csv + anomaly_label.csv + HDFS.log_templates.csv). This uses the
maintainers' Drain parse of all 11M lines, so we get the real ~29-template
sessions without re-parsing the 1.5 GB raw log.

Set the preprocessed dir via env HDFS_PRE, else the default below.
Outputs match the rest of the pipeline: sessions.parquet, split_index.json,
interim/templates.csv.
"""
import os, re, json, ast
import numpy as np
import pandas as pd
from config import PROCESSED, INTERIM, SPLIT, SEED

PRE = os.getenv("HDFS_PRE",
    os.path.expanduser("~/Downloads/HDFS_v1/preprocessed"))

SEQ_RE = re.compile(r"E\d+")

def load():
    pre = PRE
    traces = pd.read_csv(os.path.join(pre, "Event_traces.csv"),
                         usecols=["BlockId", "Features"])
    # Features look like "[E5,E22,E5,...]" -> list of EventId strings (time-ordered)
    traces["event_sequence"] = traces["Features"].map(lambda s: SEQ_RE.findall(str(s)))
    traces = traces[["BlockId", "event_sequence"]].rename(columns={"BlockId": "block_id"})
    traces["first_line"] = np.arange(len(traces))     # log order == time order proxy

    labels = pd.read_csv(os.path.join(pre, "anomaly_label.csv"))
    labels["y"] = (labels["Label"].str.strip().str.lower() == "anomaly").astype(int)
    labels = labels.rename(columns={"BlockId": "block_id"})[["block_id", "y"]]

    s = traces.merge(labels, on="block_id", how="inner")

    # chronological split by first occurrence (row order)
    s = s.sort_values("first_line").reset_index(drop=True)
    n = len(s); a, b = int(n * SPLIT[0]), int(n * (SPLIT[0] + SPLIT[1]))
    chrono = {"train": s["block_id"].iloc[:a].tolist(),
              "val":   s["block_id"].iloc[a:b].tolist(),
              "test":  s["block_id"].iloc[b:].tolist()}
    perm = s["block_id"].sample(frac=1.0, random_state=SEED).tolist()
    rnd = {"train": perm[:a], "val": perm[a:b], "test": perm[b:]}

    s.to_parquet(PROCESSED / "sessions.parquet", index=False)
    (PROCESSED / "split_index.json").write_text(
        json.dumps({"chronological": chrono, "random": rnd}))

    # official templates -> interim/templates.csv (schema: EventId, template)
    tpl = pd.read_csv(os.path.join(pre, "HDFS.log_templates.csv"))
    tpl = tpl.rename(columns={"EventTemplate": "template"})[["EventId", "template"]]
    INTERIM.mkdir(parents=True, exist_ok=True)
    tpl.to_csv(INTERIM / "templates.csv", index=False)
    return s

if __name__ == "__main__":
    s = load()
    print(f"[load_preprocessed] {len(s)} sessions, anomaly rate {100*s['y'].mean():.2f}%")
    print(f"  median/max session length = {int(s['event_sequence'].apply(len).median())}/"
          f"{int(s['event_sequence'].apply(len).max())}")
