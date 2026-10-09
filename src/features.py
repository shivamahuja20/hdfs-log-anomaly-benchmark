"""Step 5 — build the two canonical representations from sessions:
(a) event-count vectors (Tier 1)  (b) integer event sequences (Tier 2/3).
Vocabulary is fixed from the TRAIN split only (no leakage).
"""
import json, pickle
import numpy as np
import pandas as pd
from config import PROCESSED

def load():
    s = pd.read_parquet(PROCESSED / "sessions.parquet").set_index("block_id")
    split = json.loads((PROCESSED / "split_index.json").read_text())
    return s, split

def build_vocab(train_seqs):
    vocab = sorted({e for seq in train_seqs for e in seq})
    return {e: i for i, e in enumerate(vocab)}  # EventId -> column index

def count_matrix(seqs, vocab):
    M = np.zeros((len(seqs), len(vocab)), dtype=np.float32)
    for r, seq in enumerate(seqs):
        for e in seq:
            if e in vocab:
                M[r, vocab[e]] += 1
    return M

def build():
    s, split = load()
    train_ids = split["chronological"]["train"]
    vocab = build_vocab(s.loc[train_ids, "event_sequence"])
    # integer sequences keyed by block_id (unknown events -> len(vocab) = UNK)
    unk = len(vocab)
    seqs = {bid: [vocab.get(e, unk) for e in seq]
            for bid, seq in s["event_sequence"].items()}
    with (PROCESSED / "features.pkl").open("wb") as f:
        pickle.dump({"vocab": vocab, "sequences": seqs,
                     "labels": s["y"].to_dict()}, f)
    return vocab, seqs

if __name__ == "__main__":
    vocab, seqs = build()
    print(f"[features] vocab size = {len(vocab)}, sessions = {len(seqs)}")
