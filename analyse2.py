"""Uncertainty on the headline numbers, and the sentence-length effect.

0.132 comes from 500 sentences, so the post needs an interval around it before it is
compared to a leaderboard cutoff. Also pins down where the errors live.
"""

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

AUTHORS = ["EAP", "HPL", "MWS"]
EPS = 1e-15
import os
P = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)


def row_losses(y_true, probs):
    p = np.asarray(probs, dtype=float)
    p = p / p.sum(axis=1, keepdims=True)
    p = np.clip(p, EPS, 1 - EPS)
    idx = np.array([AUTHORS.index(a) for a in y_true])
    return -np.log(p[np.arange(len(idx)), idx])


def boot(losses, correct, n=20000):
    idx = rng.integers(0, len(losses), size=(n, len(losses)))
    ll = losses[idx].mean(axis=1)
    ac = correct[idx].mean(axis=1)
    return (np.percentile(ll, [2.5, 97.5]), np.percentile(ac, [2.5, 97.5]))


pred = pd.read_csv(f"{P}/predictions_claude-opus-5-5_clean_500.csv")
probs_llm = pred[AUTHORS].to_numpy()
truth_idx = np.array([AUTHORS.index(a) for a in pred.author])
correct_llm = (probs_llm.argmax(axis=1) == truth_idx).astype(float)
loss_llm = row_losses(pred.author, probs_llm)

# Baseline on exactly the same 500.
df = pd.read_csv(f"{P}/data/gh_train.csv")
train, hold = train_test_split(df, test_size=0.2, stratify=df.author, random_state=42)
word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
char = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), sublinear_tf=True, min_df=2)
X_tr = hstack([word.fit_transform(train.text), char.fit_transform(train.text)]).tocsr()
model = LogisticRegression(C=10, max_iter=2000).fit(X_tr, train.author)
order = [list(model.classes_).index(a) for a in AUTHORS]

sub = hold.set_index("id").loc[pred.id]
X_s = hstack([word.transform(sub.text), char.transform(sub.text)]).tocsr()
probs_base = model.predict_proba(X_s)[:, order]
correct_base = (probs_base.argmax(axis=1) == truth_idx).astype(float)
loss_base = row_losses(pred.author, probs_base)

print("=== headline numbers with 95% bootstrap intervals, same 500 sentences ===")
for name, losses, corr in [("LLM zero-shot", loss_llm, correct_llm),
                           ("baseline", loss_base, correct_base)]:
    (lo, hi), (alo, ahi) = boot(losses, corr)
    print(f"{name:16} log loss {losses.mean():.4f}  [{lo:.4f}, {hi:.4f}]   "
          f"accuracy {corr.mean():.4f}  [{alo:.4f}, {ahi:.4f}]")

d = loss_base - loss_llm
idx = rng.integers(0, len(d), size=(20000, len(d)))
dl, dh = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
print(f"paired difference (baseline - LLM): {d.mean():.4f}  [{dl:.4f}, {dh:.4f}]")
print()

hard = (1 - correct_llm.mean()) * -np.log(EPS)
print(f"LLM hard-label log loss: {hard:.4f}  (uniform thirds = {-np.log(1/3):.4f})")
print()

print("=== by sentence length ===")
nw = pred.text.str.split().str.len()
for label, m in [("under 20 words", nw < 20), ("20 words or more", nw >= 20)]:
    print(f"{label:18} n={m.sum():4d}  LLM accuracy {correct_llm[m.values].mean():.4f}  "
          f"LLM log loss {loss_llm[m.values].mean():.4f}  |  "
          f"baseline accuracy {correct_base[m.values].mean():.4f}  "
          f"baseline log loss {loss_base[m.values].mean():.4f}")
print()
print(f"errors: {int((1-correct_llm).sum())} of {len(pred)}; "
      f"{int((1-correct_llm)[(nw < 20).values].sum())} of them in sentences under 20 words")
print(f"median length of a wrong sentence: {nw[correct_llm == 0].median():.0f} words, "
      f"right: {nw[correct_llm == 1].median():.0f} words")
print()

print("=== the 2017 cutoffs for reference ===")
for n, v in [("1st", 0.02467), ("gold", 0.1651), ("silver", 0.2700),
             ("bronze", 0.2938), ("median", 0.4188)]:
    print(f"  {n:7} {v:.4f}")
print()

print("=== what the wrong rows look like ===")
w = pred[correct_llm == 0].copy()
w["pred"] = [AUTHORS[i] for i in probs_llm[correct_llm == 0].argmax(axis=1)]
w["conf"] = probs_llm[correct_llm == 0].max(axis=1)
w["words"] = w.text.str.split().str.len()
print(w[["author", "pred", "conf", "words", "work"]].to_string(index=False))
