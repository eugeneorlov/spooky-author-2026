"""Is "not in my corpus" a gap in Gutenberg US, or a gap in the model?

Crosses the verbatim-lookup result with the memorisation probe. If the model can recite
sentences that my corpus does not contain, then the 122 misses are my coverage problem
and the model is still doing retrieval. If it cannot recite them and still gets them
right, that part is genuine generalisation.
"""

import json
import os
import re

import numpy as np
import pandas as pd

AUTHORS = ["EAP", "HPL", "MWS"]
EPS = 1e-15
P = os.path.dirname(os.path.abspath(__file__))
CACHE = f"{P}/cache_claude-opus-5-5_clean.jsonl"
CONTINUE_HEAD = "Continue this sentence from a public-domain work of fiction"


def words(s):
    return re.findall(r"[a-z']+", s.lower())


def row_losses(y_true, probs):
    p = np.asarray(probs, dtype=float)
    p = p / p.sum(axis=1, keepdims=True)
    p = np.clip(p, EPS, 1 - EPS)
    idx = np.array([AUTHORS.index(a) for a in y_true])
    return -np.log(p[np.arange(len(idx)), idx])


pred = pd.read_csv(f"{P}/lookup_500.csv")
probs = pred[AUTHORS].to_numpy()
truth_idx = np.array([AUTHORS.index(a) for a in pred.author])
pred["correct"] = probs.argmax(axis=1) == truth_idx
pred["loss"] = row_losses(pred.author, probs)
pred["conf"] = probs.max(axis=1)
pred["words"] = pred.text.str.split().str.len()
pred["findable"] = pred.hits.fillna("").str.len() > 0
pred["named"] = ~pred.work.fillna("unknown").str.strip().str.lower().isin(
    ["unknown", "unparsed", ""])

replies = {}
with open(CACHE) as f:
    for line in f:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r["prompt"].startswith(CONTINUE_HEAD):
            replies[r["prompt"]] = r["reply"]

k = 5
memo = {}
for _, row in pred.iterrows():
    toks = row.text.split()
    if len(toks) < 20:
        continue
    cut = int(len(toks) * 0.6)
    truth = words(" ".join(toks[cut:]))[:k]
    if len(truth) < k:
        continue
    prompt = (f"Continue this sentence from a public-domain work of fiction, word for word as\n"
              f"the author wrote it. Give only the continuation, at least {k} words, nothing else.\n\n"
              f"{' '.join(toks[:cut])}")
    if prompt in replies:
        memo[row.id] = words(replies[prompt])[:25][:k] == truth
pred["memorised"] = pred.id.map(memo)

print("=== in my Gutenberg corpus, or not ===")
for label, m in [("findable by grep", pred.findable), ("not findable", ~pred.findable)]:
    s = pred[m]
    print(f"{label:<18} n={len(s):>4}  LLM accuracy {s.correct.mean():.3f}  "
          f"log loss {s.loss.mean():.4f}  named the work {s.named.mean():.1%}")
print()

print("=== the probe, split by whether my corpus has the sentence (20+ words only) ===")
probed = pred[pred.memorised.notna()]
for label, m in [("findable by grep", probed.findable), ("not findable", ~probed.findable)]:
    s = probed[m]
    print(f"{label:<18} n={len(s):>4}  recited the next 5 words {s.memorised.mean():.1%}")
print()

print("=== per author, since Gutenberg US coverage is uneven ===")
for code in AUTHORS:
    s = pred[pred.author == code]
    sp = probed[probed.author == code]
    print(f"{code}: n={len(s):>4}  findable {s.findable.mean():>6.1%}  "
          f"recited {sp.memorised.mean():>6.1%}  named {s.named.mean():>6.1%}  "
          f"LLM accuracy {s.correct.mean():.3f}  log loss {s.loss.mean():.4f}")
print()

print("=== the sentences the model got wrong that grep got right ===")
w = pred[(~pred.correct) & pred.findable].copy()
w["pred"] = [AUTHORS[i] for i in probs[((~pred.correct) & pred.findable).to_numpy()].argmax(axis=1)]
print(w[["author", "pred", "conf", "words", "hits", "work"]].to_string(index=False))
print()
print(f"model wrong on findable text: {len(w)} of {int((~pred.correct).sum())} errors")
print()

print("=== what a 2017 team could have submitted, by lookup confidence ===")
rest = lambda h: (1 - h) / 2
base = pd.read_csv(f"{P}/predictions_claude-opus-5-5_clean_500.csv")  # ids align
for hit in [0.98, 0.999, 1 - 1e-15]:
    rows = []
    for i, f in enumerate(pred.findable):
        if f:
            code = pred.hits.iloc[i]
            rows.append([hit if c == code else rest(hit) for c in AUTHORS])
        else:
            rows.append([1 / 3] * 3)
    ll = row_losses(pred.author, rows).mean()
    print(f"  {hit:<20} lookup only, thirds on a miss: log loss {ll:.4f}")
print()
print("Coverage is the binding constraint, not the method: every single hit was correct,")
print(f"so a complete corpus would drive this toward {-np.log(1 - 1e-15):.4f}.")
