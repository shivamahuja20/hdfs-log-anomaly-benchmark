"""Tier 3 — LLM classification + explanation on a STRATIFIED SAMPLE of test
sessions. Renders each session as event-template text, few-shot prompts the
model, and asks for structured JSON {label, reason} at temperature 0.

Backends:
  * "openai" : real GPT-4o-mini. Needs OPENAI_API_KEY env var and `pip install openai`.
  * "mock"   : offline heuristic used to validate prompt-building / parsing / scoring
               without an API key. Clearly NOT a model result.
Every API response is cached to results/llm_cache.json so you never pay twice.
"""
import os, json, pickle, random, hashlib
import pandas as pd
from config import PROCESSED, INTERIM, RESULTS, LLM_MODEL, LLM_SAMPLE, FEWSHOT_K, SEED
from evaluate import evaluate

CACHE = RESULTS / "llm_cache.json"

SYSTEM = (
    "You are an SRE assistant analysing HDFS block lifecycle logs. A NORMAL block "
    "is allocated, its data is received by each replica, packet responders "
    "acknowledge, the block is added to the block map, verified, and optionally "
    "served or deleted. ANOMALOUS sessions show broken sequences: missing "
    "acknowledgements, exceptions, replication timeouts, reordering, or redundant "
    "events. Given a session's ordered event templates, decide if it is Normal or "
    "Anomalous and give a one-sentence reason. Respond ONLY as JSON: "
    '{"label": "Normal"|"Anomalous", "reason": "..."}.'
)

def _id2text():
    tpl = pd.read_csv(INTERIM / "templates.csv").set_index("EventId")["template"].to_dict()
    with (PROCESSED / "features.pkl").open("rb") as f:
        vocab = pickle.load(f)["vocab"]
    inv = {v: k for k, v in vocab.items()}       # int -> EventId
    return lambda seq: " -> ".join(tpl.get(inv.get(e, "?"), f"E{e}") for e in seq)

def _sample(split_name="chronological"):
    with (PROCESSED / "features.pkl").open("rb") as f:
        F = pickle.load(f)
    split = json.loads((PROCESSED / "split_index.json").read_text())[split_name]
    y = F["labels"]; test = split["test"]
    anom = [b for b in test if y[b] == 1]
    norm = [b for b in test if y[b] == 0]
    random.seed(SEED)
    k = LLM_SAMPLE // 2
    samp = random.sample(anom, min(k, len(anom))) + random.sample(norm, min(k, len(norm)))
    random.shuffle(samp)
    return F, split_name, samp, split["train"]

def _fewshot(F, train_ids, render):
    y = F["labels"]; seqs = F["sequences"]
    random.seed(SEED)
    a = random.sample([b for b in train_ids if y[b] == 1], FEWSHOT_K)
    n = random.sample([b for b in train_ids if y[b] == 0], FEWSHOT_K)
    ex = []
    for b in a + n:
        ex.append({"role": "user", "content": "Session: " + render(seqs[b])})
        ex.append({"role": "assistant",
                   "content": json.dumps({"label": "Anomalous" if y[b] else "Normal",
                                          "reason": "example"})})
    return ex

def _mock_classify(text):
    # offline stand-in ONLY: flags explicit error phrases or unusually short flows.
    if any(w in text for w in ["exception", "timed out", "Redundant"]):
        return {"label": "Anomalous", "reason": "mock: error/timeout/redundant event present"}
    if text.count("->") < 6:
        return {"label": "Anomalous", "reason": "mock: unusually short lifecycle"}
    return {"label": "Normal", "reason": "mock: lifecycle looks complete"}

def run(split_name="chronological", backend=None):
    F, split_name, samp, train_ids = _sample(split_name)
    render = _id2text()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    backend = backend or ("openai" if os.getenv("OPENAI_API_KEY") else "mock")

    client = None
    if backend == "openai":
        from openai import OpenAI
        client = OpenAI()
        fewshot = _fewshot(F, train_ids, render)

    y_true, y_pred, explanations, tokens = [], [], [], 0
    for b in samp:
        text = render(F["sequences"][b])
        key = hashlib.md5((backend + text).encode()).hexdigest()
        if key in cache:
            out = cache[key]
        elif backend == "openai":
            msgs = [{"role": "system", "content": SYSTEM}] + fewshot + \
                   [{"role": "user", "content": "Session: " + text}]
            resp = client.chat.completions.create(
                model=LLM_MODEL, messages=msgs, temperature=0,
                response_format={"type": "json_object"})
            out = json.loads(resp.choices[0].message.content)
            tokens += resp.usage.total_tokens
            cache[key] = out
        else:
            out = _mock_classify(text); cache[key] = out
        y_true.append(int(F["labels"][b]))
        y_pred.append(1 if str(out.get("label", "")).lower().startswith("anom") else 0)
        explanations.append({"block_id": b, "label": out.get("label"), "reason": out.get("reason")})

    CACHE.write_text(json.dumps(cache))
    (RESULTS / "llm_explanations.json").write_text(json.dumps(explanations, indent=2))
    res = evaluate(y_true, y_pred, f"Tier3-LLM({backend})", "text", split_name)
    res["backend"] = backend
    res["approx_tokens"] = tokens
    return res

if __name__ == "__main__":
    print(run())
