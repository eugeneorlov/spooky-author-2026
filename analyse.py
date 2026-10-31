"""Where the LLM's score actually comes from.

Joins three things per sentence: the zero-shot probabilities, whether the model named the
work, and whether it could reproduce the next five words of the real text. If recognition
is doing the work, accuracy should track memorisation rather than style.
"""

import json
import re
import sys

import numpy as np
import pandas as pd

AUTHORS = ["EAP", "HPL", "MWS"]
EPS = 1e-15
import os
P = os.path.dirname(os.path.abspath(__file__))
PRED = f"{P}/predictions_claude-opus-5-5_clean_500.csv"
CACHE = f"{P}/cache_claude-opus-5-5_clean.jsonl"

CONTINUE_HEAD = "Continue this sentence from a public-domain work of fiction"


def words(s):
    return re.findall(r"[a-z']+", s.lower())


def log_loss(y_true, probs):
    p = np.asarray(probs, dtype=float)
    p = p / p.sum(axis=1, keepdims=True)
    p = np.clip(p, EPS, 1 - EPS)
    idx = np.array([AUTHORS.index(a) for a in y_true])
    return float(-np.mean(np.log(p[np.arange(len(idx)), idx])))


pred = pd.read_csv(PRED)
probs = pred[AUTHORS].to_numpy()
truth_idx = np.array([AUTHORS.index(a) for a in pred.author])
pred["correct"] = probs.argmax(axis=1) == truth_idx
pred["p_true"] = probs[np.arange(len(probs)), truth_idx]
pred["top"] = probs.max(axis=1)
pred["named"] = ~pred.work.fillna("unknown").str.strip().str.lower().isin(
    ["unknown", "unparsed", ""])

# Rebuild the probe outcome per sentence from the cached continuation replies.
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
    prefix = " ".join(toks[:cut])
    truth = words(" ".join(toks[cut:]))[:k]
    if len(truth) < k:
        continue
    prompt = (f"Continue this sentence from a public-domain work of fiction, word for word as\n"
              f"the author wrote it. Give only the continuation, at least {k} words, nothing else.\n\n"
              f"{prefix}")
    if prompt not in replies:
        continue
    got = words(replies[prompt])[:25]
    memo[row.id] = got[:k] == truth

pred["memorised"] = pred.id.map(memo)
probed = pred[pred.memorised.notna()].copy()
probed["memorised"] = probed.memorised.astype(bool)

print(f"sentences classified: {len(pred)}")
print(f"overall log loss {log_loss(pred.author, probs):.4f}   accuracy {pred.correct.mean():.4f}")
print(f"hard-label log loss {(1 - pred.correct.mean()) * -np.log(EPS):.4f}")
print()

print("=== did it name the work ===")
for flag, label in [(True, "named"), (False, "did not name")]:
    g = pred[pred.named == flag]
    print(f"{label:14} n={len(g):4d} ({len(g)/len(pred):5.1%})  accuracy {g.correct.mean():.3f}  "
          f"log loss {log_loss(g.author, g[AUTHORS].to_numpy()):.3f}  mean confidence {g.top.mean():.3f}")
print()

print("=== could it reproduce the next 5 words (sentences with 20+ words) ===")
print(f"probed: {len(probed)}   memorised: {probed.memorised.mean():.1%}")
for flag, label in [(True, "memorised"), (False, "not memorised")]:
    g = probed[probed.memorised == flag]
    if not len(g):
        continue
    print(f"{label:14} n={len(g):4d} ({len(g)/len(probed):5.1%})  accuracy {g.correct.mean():.3f}  "
          f"log loss {log_loss(g.author, g[AUTHORS].to_numpy()):.3f}  mean confidence {g.top.mean():.3f}")
print()

print("=== memorisation against naming ===")
print(pd.crosstab(probed.memorised, probed.named, margins=True).to_string())
print()
print("accuracy by cell:")
print(probed.groupby(["memorised", "named"]).correct.agg(["mean", "size"]).to_string())
print()

print("=== LLM calibration ===")
for lo, hi in [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.98), (0.98, 1.01)]:
    m = (pred.top >= lo) & (pred.top < hi)
    if m.sum():
        print(f"  top in [{lo:.2f},{hi:.2f}): n={m.sum():4d}  mean confidence {pred.top[m].mean():.3f}  "
              f"actual accuracy {pred.correct[m].mean():.3f}")
print()
print(f"rows where it gave the true author under 0.01: {(pred.p_true < 0.01).sum()}")
print(f"worst single-row loss: {-np.log(max(pred.p_true.min(), EPS)):.2f}")
print(f"share of total loss from the {(~pred.correct).sum()} wrong rows: "
      f"{(-np.log(np.clip(pred.p_true[~pred.correct], EPS, 1))).sum() / (-np.log(np.clip(pred.p_true, EPS, 1))).sum():.1%}")
print()

print("=== most-named works ===")
print(pred[pred.named].work.str.strip().value_counts().head(12).to_string())
