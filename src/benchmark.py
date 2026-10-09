"""Runs every tier through the shared evaluator, assembles results.csv, runs the
chronological-vs-random split ablation, and renders the report figures.
"""
import warnings; warnings.filterwarnings("ignore")
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import RESULTS, FIGS
import tier0_rules, tier1_classical, tier2_deeplog, tier3_llm

def _safe(rows, fn, *a, label=""):
    """Run a tier; never let one tier's failure discard the others."""
    try:
        r = fn(*a)
        rows += r if isinstance(r, list) else [r]
    except Exception as e:
        print(f"  [skipped {label}] {type(e).__name__}: {str(e)[:200]}")

def collect():
    rows = []
    # --- primary: chronological split ---
    _safe(rows, tier0_rules.run, "chronological", label="Tier0")
    _safe(rows, tier1_classical.run, "chronological", label="Tier1")
    _safe(rows, tier2_deeplog.run, "chronological", label="Tier2")
    # save what we have BEFORE the network-dependent LLM tier, so a bad key
    # can never cost us the LSTM result
    pd.DataFrame(rows).to_csv(RESULTS / "results_partial.csv", index=False)
    _safe(rows, tier3_llm.run, "chronological", label="Tier3-LLM")
    # --- ablation: random split (Le & Zhang split-sensitivity) ---
    _safe(rows, tier1_classical.run, "random", label="Tier1-random")
    _safe(rows, tier2_deeplog.run, "random", label="Tier2-random")
    df = pd.DataFrame(rows)
    keep = ["tier", "feature", "split", "precision", "recall", "f1", "fpr",
            "tp", "fp", "fn", "tn"]
    df = df[[c for c in keep if c in df.columns] +
            [c for c in df.columns if c not in keep]]
    df.to_csv(RESULTS / "results.csv", index=False)
    return df

def fig_f1(df):
    d = df[df["split"] == "chronological"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(d))
    ax.bar([i - 0.2 for i in x], d["f1"], 0.4, label="F1", color="#2E5496")
    ax.bar([i + 0.2 for i in x], d["fpr"], 0.4, label="FPR", color="#C45911")
    ax.set_xticks(list(x)); ax.set_xticklabels(d["tier"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("score"); ax.set_title("Model comparison (chronological split)")
    ax.legend(); ax.grid(axis="y", alpha=.3); fig.tight_layout()
    fig.savefig(FIGS / "model_comparison.png", dpi=140); plt.close(fig)

def fig_confusion(df):
    d = df[df["split"] == "chronological"].reset_index(drop=True)
    n = len(d); cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.8 * rows))
    for i, (_, r) in enumerate(d.iterrows()):
        ax = axes.ravel()[i]
        cm = [[r["tn"], r["fp"]], [r["fn"], r["tp"]]]
        ax.imshow(cm, cmap="Blues")
        for (a, b), v in [((0,0),cm[0][0]),((0,1),cm[0][1]),((1,0),cm[1][0]),((1,1),cm[1][1])]:
            ax.text(b, a, str(v), ha="center", va="center", fontsize=10)
        ax.set_title(r["tier"], fontsize=8)
        ax.set_xticks([0,1]); ax.set_xticklabels(["N","A"], fontsize=7)
        ax.set_yticks([0,1]); ax.set_yticklabels(["N","A"], fontsize=7)
        ax.set_xlabel("pred", fontsize=7); ax.set_ylabel("true", fontsize=7)
    for j in range(n, rows*cols): axes.ravel()[j].axis("off")
    fig.suptitle("Confusion matrices (chronological test set)")
    fig.tight_layout(); fig.savefig(FIGS / "confusion_matrices.png", dpi=140); plt.close(fig)

def fig_split(df):
    piv = df.pivot_table(index="tier", columns="split", values="f1")
    if "random" not in piv.columns: return
    piv = piv.dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    piv.plot(kind="bar", ax=ax, color={"chronological": "#2E5496", "random": "#9DC3E6"})
    ax.set_ylabel("F1"); ax.set_title("Split sensitivity: chronological vs random")
    ax.set_xticklabels(piv.index, rotation=20, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=.3); fig.tight_layout()
    fig.savefig(FIGS / "split_sensitivity.png", dpi=140); plt.close(fig)

if __name__ == "__main__":
    df = collect()
    fig_f1(df); fig_confusion(df); fig_split(df)
    print(df.to_string(index=False))
    print("\nfigures ->", FIGS)
