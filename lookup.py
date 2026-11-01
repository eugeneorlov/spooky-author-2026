"""What could a 2017 participant have scored with the original books and grep?

The competition rules allowed external data. All three authors are on Project Gutenberg.
So the question is whether the held-out sentences can be found verbatim in the books,
which turns the task from classification into a table lookup.

Scored three ways on the same 500 sentences the LLM saw:
  1. lookup only, uniform thirds where no match is found
  2. lookup, with the 2017 TF-IDF baseline filling in the misses
  3. the LLM, for comparison (read from its predictions file)
"""

import os

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import corpus
from corpus import AUTHORS, norm

P = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-15
HIT = 0.98  # what a sane participant would submit on a verbatim match, not 1.0


def row_losses(y_true, probs):
    p = np.asarray(probs, dtype=float)
    p = p / p.sum(axis=1, keepdims=True)
    p = np.clip(p, EPS, 1 - EPS)
    idx = np.array([AUTHORS.index(a) for a in y_true])
    return -np.log(p[np.arange(len(idx)), idx])


def report(name, y_true, probs):
    losses = row_losses(y_true, probs)
    idx = np.array([AUTHORS.index(a) for a in y_true])
    acc = (np.asarray(probs).argmax(axis=1) == idx).mean()
    print(f"{name:<44}{losses.mean():>9.4f}{acc:>10.3f}")
    return losses


pred = pd.read_csv(f"{P}/predictions_claude-opus-5-5_clean_500.csv")
print("=== corpus ===")
hay = {}
for code in AUTHORS:
    h, n_files = corpus.haystack(code)
    hay[code] = h
    print(f"{code}: {n_files} files, {len(h)/1e6:.1f} MB normalised")
print()

# 1. Does the sentence appear verbatim in each author's books?
print("=== verbatim lookup of the 500 held-out sentences ===")
found = []
for text in pred.text:
    n = norm(text)
    hits = [code for code in AUTHORS if f" {n} " in hay[code]]
    found.append(hits)

pred["hits"] = ["+".join(h) if h else "" for h in found]
n_one = sum(1 for h in found if len(h) == 1)
n_many = sum(1 for h in found if len(h) > 1)
n_none = sum(1 for h in found if not h)
print(f"found in exactly one author's books: {n_one} ({n_one/len(pred):.1%})")
print(f"found in more than one:              {n_many}")
print(f"not found:                           {n_none} ({n_none/len(pred):.1%})")

single = [(h[0] if len(h) == 1 else None) for h in found]
agree = sum(1 for a, s in zip(pred.author, single) if s is not None and s == a)
print(f"of the single hits, the author matched the label: {agree}/{n_one}")
print()

print("=== coverage by true author ===")
for code in AUTHORS:
    m = pred.author == code
    hit = [bool(h) for h, keep in zip(found, m) if keep]
    print(f"{code}: {sum(hit)}/{len(hit)} sentences found ({sum(hit)/len(hit):.1%})")
print()

print("=== by sentence length ===")
nw = pred.text.str.split().str.len()
for label, m in [("under 20 words", nw < 20), ("20 words or more", nw >= 20)]:
    hit = [bool(h) for h, keep in zip(found, m) if keep]
    print(f"{label:18} {sum(hit)}/{len(hit)} found ({sum(hit)/len(hit):.1%})")
print()

# 2. The 2017 baseline, for the fallback and for reference.
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

rest = (1 - HIT) / 2
probs_lookup_uniform, probs_lookup_base = [], []
for i, h in enumerate(found):
    if len(h) == 1:
        row = [HIT if c == h[0] else rest for c in AUTHORS]
        probs_lookup_uniform.append(row)
        probs_lookup_base.append(row)
    else:
        probs_lookup_uniform.append([1 / 3] * 3)
        probs_lookup_base.append(list(probs_base[i]))

print(f"{'predictor':<44}{'log loss':>9}{'accuracy':>10}")
report("2017 recipe on train.csv (same 500)", pred.author, probs_base)
report("LLM zero-shot (same 500)", pred.author, pred[AUTHORS].to_numpy())
report(f"lookup only, {HIT} on a hit, thirds otherwise", pred.author, probs_lookup_uniform)
report("lookup, 2017 baseline fills the misses", pred.author, probs_lookup_base)
print()

print("=== where the lookup and the LLM disagree with the truth ===")
llm_pred = [AUTHORS[i] for i in pred[AUTHORS].to_numpy().argmax(axis=1)]
llm_wrong = [a != b for a, b in zip(pred.author, llm_pred)]
lk_found = [bool(h) for h in found]
print(f"LLM wrong and sentence was findable:     "
      f"{sum(1 for w, f in zip(llm_wrong, lk_found) if w and f)}")
print(f"LLM wrong and sentence was not findable: "
      f"{sum(1 for w, f in zip(llm_wrong, lk_found) if w and not f)}")
print(f"LLM right but sentence not findable:     "
      f"{sum(1 for w, f in zip(llm_wrong, lk_found) if not w and not f)}")
pred.to_csv(f"{P}/lookup_500.csv", index=False)
