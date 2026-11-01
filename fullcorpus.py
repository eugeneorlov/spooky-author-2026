"""If a 2017 team had trained the same model on the authors' complete works, how far
would that have got them?

Three training sets, one model class (the 2017 recipe), one evaluation: the same 500
sentences the LLM saw, plus the full 3,916-sentence holdout.

  A  corpus only, with every holdout sentence removed      -> does more in-domain data help?
  B  corpus + the 80% train split, holdout removed         -> the honest 2017 maximum
  C  corpus + train split, holdout sentences left in       -> what training on the books
                                                              gives you for free, which is
                                                              a weak form of lookup
"""

import argparse
import os
import random
import time

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

ap = argparse.ArgumentParser()
ap.add_argument("--cap", type=int, default=30000, help="max corpus sentences per author")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
random.seed(args.seed)


def score(y_true, probs):
    p = np.asarray(probs, dtype=float)
    p = p / p.sum(axis=1, keepdims=True)
    p = np.clip(p, EPS, 1 - EPS)
    idx = np.array([AUTHORS.index(a) for a in y_true])
    ll = float(-np.mean(np.log(p[np.arange(len(idx)), idx])))
    acc = float((p.argmax(axis=1) == idx).mean())
    return ll, acc


def fit_and_score(label, texts, labels, evals):
    t0 = time.time()
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), sublinear_tf=True, min_df=2)
    X = hstack([word.fit_transform(texts), char.fit_transform(texts)]).tocsr()
    model = LogisticRegression(C=10, max_iter=2000).fit(X, labels)
    order = [list(model.classes_).index(a) for a in AUTHORS]
    fit_s = time.time() - t0
    print(f"\n--- {label} ---")
    print(f"training sentences: {len(texts):,}   features: {X.shape[1]:,}   fit {fit_s:.0f}s")
    for ev_name, ev_texts, ev_true in evals:
        Xe = hstack([word.transform(ev_texts), char.transform(ev_texts)]).tocsr()
        probs = model.predict_proba(Xe)[:, order]
        ll, acc = score(ev_true, probs)
        print(f"  {ev_name:<28} log loss {ll:.4f}   accuracy {acc:.4f}")


# Evaluation sets: the LLM's 500, and the full holdout.
df = pd.read_csv(f"{P}/data/gh_train.csv")
train, hold = train_test_split(df, test_size=0.2, stratify=df.author, random_state=42)
pred = pd.read_csv(f"{P}/predictions_claude-opus-5-5_clean_500.csv")
sub = hold.set_index("id").loc[pred.id]
evals = [("the LLM's 500", sub.text.tolist(), sub.author.tolist()),
         ("full holdout (3,916)", hold.text.tolist(), hold.author.tolist())]

# Every holdout sentence, normalised, so corpus copies of them can be removed.
hold_norm = set(norm(t) for t in hold.text)

print("=== corpus sentences ===")
corpus_sents, corpus_labels, corpus_sents_dirty, corpus_labels_dirty = [], [], [], []
for code in AUTHORS:
    sents, n_files = corpus.sentences(code)
    seen, clean, dirty = set(), [], []
    for s in sents:
        n = norm(s)
        if not n or n in seen:
            continue
        seen.add(n)
        dirty.append(s)
        if n not in hold_norm:
            clean.append(s)
    removed = len(dirty) - len(clean)
    random.shuffle(clean)
    random.shuffle(dirty)
    clean, dirty = clean[:args.cap], dirty[:args.cap]
    print(f"{code}: {n_files} files, {len(sents):,} raw, {len(seen):,} unique, "
          f"{removed:,} were holdout sentences, using {len(clean):,}")
    corpus_sents += clean
    corpus_labels += [code] * len(clean)
    corpus_sents_dirty += dirty
    corpus_labels_dirty += [code] * len(dirty)

print(f"\nbaseline for comparison: the 2017 recipe on {len(train):,} train.csv sentences")
fit_and_score("baseline: train.csv 80% split", train.text.tolist(), train.author.tolist(), evals)
fit_and_score("A: corpus only, holdout removed", corpus_sents, corpus_labels, evals)
fit_and_score("B: corpus + train split, holdout removed",
              corpus_sents + train.text.tolist(), corpus_labels + train.author.tolist(), evals)
fit_and_score("C: corpus + train split, holdout left in",
              corpus_sents_dirty + train.text.tolist(),
              corpus_labels_dirty + train.author.tolist(), evals)
