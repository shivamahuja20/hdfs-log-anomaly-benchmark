"""CLASSROOM DEMO — runs in seconds, no network / API key / training needed.
Part 1: the real benchmark results.
Part 2: real HDFS sessions run through ALL FOUR TIERS side-by-side, so the class
        sees which approaches catch each anomaly and which miss it.

Prereq (run ONCE, tonight):  python src/make_demo_data.py   (builds predictions.csv)
Then in class:               python src/demo.py
"""
import json, time, textwrap, pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"; PROC = ROOT / "data" / "processed"; INT = ROOT / "data" / "interim"
BAR = "=" * 78
def hr(c="-"): print(c * 78)
def pause(s=0.8): time.sleep(s)
def short(t): return " ".join(t.replace("[*]", " ").split())[:44]
def render_seq(seq, tmap, n=10):
    names = [short(tmap.get(e, e)) for e in seq]
    out = "  ->  ".join(names[:n])
    if len(names) > n: out += f"   ... (+{len(names)-n} more, {len(names)} total)"
    return out

TIERS = [  # (label, column, note)
    ("Tier 0  Keyword rules  (industry baseline)", "tier0", ""),
    ("Tier 1  PCA            (classical ML)",      "tier1_pca", ""),
    ("Tier 1  Isolation Forest (classical ML)",   "tier1_iforest", ""),
    ("Tier 2  DeepLog LSTM   (our detector)",      "tier2", "<<"),
    ("Tier 3  GPT-4o-mini    (explains)",          "tier3", ""),
]

def verdict_block(row):
    true = int(row["y_true"])
    for label, col, note in TIERS:
        v = row.get(col)
        if pd.isna(v):
            line = f"   {label:<44}  (n/a)"
        else:
            pred = int(v)
            word = "ANOMALOUS" if pred else "NORMAL"
            tag = "" if pred == true else "   <-- MISS"
            star = "   " + note if note else ""
            line = f"   {label:<44}{word:<10}{tag}{star}"
        print(line)
        pause(0.25)

def main():
    print("\n" + BAR)
    print("  DETECTING ANOMALOUS BACKEND LOG SESSIONS  —  Group 10 (A10)")
    print("  One HDFS session  ->  four AI approaches  ->  who catches it?")
    print(BAR + "\n"); pause()

    # ---------- Part 1 ----------
    print(">> PART 1  —  How well does each approach work?  (real HDFS_v1)\n")
    df = pd.read_csv(RES / "results.csv")
    d = df[df["split"] == "chronological"]
    name = {"Tier0-Rules":"Keyword rules (industry baseline)","Tier1-PCA":"Classical ML - PCA",
            "Tier1-iForest":"Classical ML - Isolation Forest","Tier2-DeepLog(lstm)":"DeepLog LSTM (our detector)",
            "Tier3-LLM(openai)":"GPT-4o-mini (explanation tier)*"}
    print(f"   {'Approach':<38}{'F1':>7}{'Recall':>9}{'FalseAlarm':>12}"); hr()
    for _, r in d.iterrows():
        print(f"   {name.get(r['tier'], r['tier']):<38}{r['f1']:>7.3f}{r['recall']:>9.2f}{r['fpr']*100:>11.2f}%")
    hr()
    print("   * LLM scored on a balanced sample; used mainly to EXPLAIN flags.")
    print("   Best live detector: DeepLog LSTM — F1 0.89 at 0.04% false alarms.\n"); pause(1.2)

    # ---------- Part 2 ----------
    print(BAR)
    print(">> PART 2  —  Same session, all four tiers.  Who catches the anomaly?\n"); pause()
    if not (RES / "predictions.csv").exists():
        print("   [!] results/predictions.csv not found.")
        print("       Run once first:   python src/make_demo_data.py")
        print("\n" + BAR + "\n"); return
    pred = pd.read_csv(RES / "predictions.csv").set_index("block_id")
    sessions = pd.read_parquet(PROC / "sessions.parquet").set_index("block_id")
    tmap = pd.read_csv(INT / "templates.csv").set_index("EventId")["template"].to_dict()
    samp = pred[pred["tier3"].notna()]

    # Example 1: TRUE anomaly the keyword rules MISS but our model catches (the money shot)
    a1 = samp[(samp.y_true == 1) & (samp.tier0 == 0) & (samp.tier2 == 1) & (samp.tier3 == 1)
              & (samp.tier3_reason.str.len() > 25)]
    # Example 2: a clean NORMAL all tiers agree on
    nrm = samp[(samp.y_true == 0) & (samp.tier0 == 0) & (samp.tier1_pca == 0) & (samp.tier2 == 0)]
    # Example 3: another anomaly with a DIFFERENT reason
    used = set(a1["tier3_reason"].head(1).str[:40].str.lower())
    a2 = samp[(samp.y_true == 1) & (samp.tier2 == 1) & (samp.tier3 == 1)
              & (~samp.tier3_reason.str[:40].str.lower().isin(used))
              & (samp.tier3_reason.str.len() > 25)]
    picks = [("Anomaly the industry method MISSES", a1.index[0]),
             ("A healthy session (no false alarm)", nrm.index[0]),
             ("Another anomaly, different failure", a2.index[0])]

    for i, (title, b) in enumerate(picks, 1):
        row = pred.loc[b]
        print(f"\n   ---- Example {i}: {title} ----")
        print(f"   session {b}    GROUND TRUTH: "
              f"{'ANOMALOUS' if int(row.y_true) else 'NORMAL'}")
        pause(0.4)
        seq = list(sessions.loc[b, "event_sequence"])
        print("   log events:")
        print(textwrap.fill(render_seq(seq, tmap), 72, initial_indent="     ", subsequent_indent="     "))
        print()
        pause(0.5)
        verdict_block(row)
        if int(row.get("tier3", 0) or 0) == 1 and isinstance(row.get("tier3_reason"), str):
            print(textwrap.fill('   GPT-4o-mini reason: "' + row["tier3_reason"].strip() + '"',
                                74, subsequent_indent="        "))
        hr(); pause(1.0)

    print("\n   THE STORY:  keyword rules and tree-based ML miss silent failures;")
    print("   the order-aware DeepLog LSTM catches them at ~zero false alarms; the")
    print("   LLM adds the plain-English reason. That is why we recommend both.\n")
    print(BAR + "\n")

if __name__ == "__main__":
    main()
