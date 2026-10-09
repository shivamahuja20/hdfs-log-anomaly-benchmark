"""Tier 2 — DeepLog. Train a next-event predictor on NORMAL sequences only;
flag a session if, at any window, the true next event is NOT in the model's
top-g predictions.

Two interchangeable backends behind ONE detection/eval path:
  * backend="lstm"   -> the real DeepLog LSTM (PyTorch). Use for the final run.
  * backend="markov" -> a dependency-free next-event model (NumPy) that runs
                        anywhere; used to validate the pipeline when torch is
                        unavailable. The windowing, top-g rule and scoring are
                        identical, so results are directly comparable in kind.
The backend auto-selects: LSTM if torch is importable, else Markov.
"""
import json, pickle
import numpy as np
import pandas as pd
from config import (PROCESSED, MODELS, WINDOW_H, TOPG, LSTM_LAYERS, LSTM_HIDDEN,
                    BATCH, EPOCHS, SEED)
try:
    from config import MAX_TRAIN_WINDOWS
except Exception:
    MAX_TRAIN_WINDOWS = None
from evaluate import evaluate

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


# ---------- shared data prep ----------
def _load():
    with (PROCESSED / "features.pkl").open("rb") as f:
        F = pickle.load(f)
    s = pd.read_parquet(PROCESSED / "sessions.parquet").set_index("block_id")
    split = json.loads((PROCESSED / "split_index.json").read_text())
    return F, s, split

def _windows(seqs, h):
    """Yield (context[h], next) training pairs from a list of int-sequences."""
    X, y = [], []
    for seq in seqs:
        if len(seq) <= h:
            pad = [0] * (h - len(seq) + 1) + seq        # left-pad short sessions
            seq = pad
        for i in range(len(seq) - h):
            X.append(seq[i:i + h]); y.append(seq[i + h])
    return np.array(X, dtype=np.int64), np.array(y, dtype=np.int64)


# ---------- LSTM backend (real DeepLog) ----------
class _LSTM(nn.Module if HAS_TORCH else object):
    def __init__(self, vocab, hidden, layers):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.lstm = nn.LSTM(hidden, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, vocab)
    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        return self.fc(h[:, -1, :])

def _train_lstm(Xtr, ytr, vocab):
    import time
    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = _LSTM(vocab, LSTM_HIDDEN, LSTM_LAYERS).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    nb = (len(Xt) + BATCH - 1) // BATCH
    print(f"    training LSTM on {len(Xt):,} windows, {EPOCHS} epochs "
          f"({nb} batches/epoch) on {dev} ...", flush=True)
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(len(Xt)); running = 0.0; t0 = time.time()
        for i in range(0, len(Xt), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            out = net(Xt[idx].to(dev))
            loss = lossf(out, yt[idx].to(dev))
            loss.backward(); opt.step()
            running += loss.item()
        print(f"    epoch {ep+1:2d}/{EPOCHS}  loss={running/nb:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    net.eval()
    return net, dev

def _topg_lstm(net, dev, contexts, g):
    with torch.no_grad():
        logits = net(torch.tensor(contexts).to(dev))
        topg = torch.topk(logits, g, dim=1).indices.cpu().numpy()
    return topg


# ---------- Markov backend (validation fallback) ----------
def _train_markov(Xtr, ytr, vocab):
    # P(next | last event in context); Laplace-smoothed counts
    counts = np.ones((vocab, vocab), dtype=np.float64)
    for ctx, nxt in zip(Xtr, ytr):
        counts[ctx[-1], nxt] += 1
    return counts / counts.sum(1, keepdims=True)

def _topg_markov(probs, contexts, g):
    last = contexts[:, -1]
    return np.argsort(-probs[last], axis=1)[:, :g]


# ---------- unified detection ----------
def run(split_name="chronological", backend=None, g=TOPG, h=WINDOW_H):
    F, s, split = _load()
    vocab = len(F["vocab"]) + 1                     # +1 for UNK
    seqs = F["sequences"]
    sp = split[split_name]
    train_ids, test_ids = sp["train"], sp["test"]
    y = F["labels"]

    train_norm = [seqs[b] for b in train_ids if y[b] == 0]   # normal only
    Xtr, ytr = _windows(train_norm, h)
    # subsample training windows for CPU speed (LSTM only; Markov uses all)
    if MAX_TRAIN_WINDOWS and len(Xtr) > MAX_TRAIN_WINDOWS:
        rng = np.random.default_rng(SEED)
        sel = rng.choice(len(Xtr), MAX_TRAIN_WINDOWS, replace=False)
        Xtr, ytr = Xtr[sel], ytr[sel]

    backend = backend or ("lstm" if HAS_TORCH else "markov")
    if backend == "lstm":
        model, dev = _train_lstm(Xtr, ytr, vocab)
        topg_fn = lambda ctx: _topg_lstm(model, dev, ctx, g)
    else:
        model = _train_markov(Xtr, ytr, vocab)
        topg_fn = lambda ctx: _topg_markov(model, ctx, g)

    def predict(ids):
        # Collect ALL windows across sessions, score in big batches, then reduce
        # per session (a session is anomalous if ANY window violates top-g).
        ctxs, nxts, owner = [], [], []
        for si, b in enumerate(ids):
            seq = seqs[b]
            seq_p = [0] * (h - len(seq) + 1) + seq if len(seq) <= h else seq
            for i in range(len(seq_p) - h):
                ctxs.append(seq_p[i:i + h]); nxts.append(seq_p[i + h]); owner.append(si)
        yhat = [0] * len(ids)
        if not ctxs:
            return yhat
        ctxs = np.asarray(ctxs, dtype=np.int64)
        nxts = np.asarray(nxts, dtype=np.int64)
        owner = np.asarray(owner)
        viol = np.zeros(len(ctxs), dtype=bool)
        CH = 20000
        for s in range(0, len(ctxs), CH):
            tg = topg_fn(ctxs[s:s + CH])
            in_topg = (tg == nxts[s:s + CH, None]).any(axis=1)
            viol[s:s + CH] = ~in_topg
        # any violating window -> session anomalous
        flagged = np.unique(owner[viol])
        for si in flagged:
            yhat[si] = 1
        return yhat

    yte = [int(y[b]) for b in test_ids]
    yhat = predict(test_ids)
    with (MODELS / f"tier2_{backend}.pkl").open("wb") as f:
        pickle.dump({"backend": backend, "g": g, "h": h}, f)
    # export per-session predictions (used by the demo's tier-by-tier comparison)
    import pandas as _pd
    from config import RESULTS as _R
    _pd.DataFrame({"block_id": test_ids, "y_true": yte, "tier2": yhat}) \
        .to_csv(_R / f"pred_tier2_{split_name}.csv", index=False)
    res = evaluate(yte, yhat, f"Tier2-DeepLog({backend})", "sequence", split_name)
    res["backend"] = backend
    return res

if __name__ == "__main__":
    print(f"[tier2] torch available: {HAS_TORCH}")
    print(run())
