"""Tier 0 — keyword rule baseline (the traditional-practice control).
A session is Anomalous if ANY of its event templates contains an error keyword.
"""
import re, json
import pandas as pd
from config import PROCESSED, INTERIM
from evaluate import evaluate

KW = re.compile(r"error|exception|failed|fatal|timed out|received exception", re.I)

def predict(split_name="chronological"):
    s = pd.read_parquet(PROCESSED / "sessions.parquet").set_index("block_id")
    tpl = pd.read_csv(INTERIM / "templates.csv")
    bad_events = set(tpl.loc[tpl["template"].str.contains(KW), "EventId"])
    split = json.loads((PROCESSED / "split_index.json").read_text())[split_name]
    test = split["test"]
    y_true = [int(s.loc[b, "y"]) for b in test]
    y_pred = [int(any(e in bad_events for e in s.loc[b, "event_sequence"])) for b in test]
    return y_true, y_pred

def run(split_name="chronological"):
    yt, yp = predict(split_name)
    return evaluate(yt, yp, "Tier0-Rules", "keywords", split_name)

if __name__ == "__main__":
    print(run())
