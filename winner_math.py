"""What corpus coverage does each 2017 leaderboard score imply, if the method was a lookup?

The lookup experiment measured both terms that matter: the cost of a verbatim hit, and
the 2017 baseline's cost on the rows a lookup misses. That gives a one-line model of
score against coverage, which the measured run validates at 75.6% coverage.
"""

import os

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

AUTHORS = ["EAP", "HPL", "MWS"]
EPS = 1e-15
P = os.path.dirname(os.path.abspath(__file__))

pred = pd.read_csv(f"{P}/lookup_500.csv")
findable = (pred.hits.fillna("").str.len() > 0).to_numpy()

df = pd.read_csv(f"{P}/data/gh_train.csv")
train, hold = train_test_split(df, test_size=0.2, stratify=df.author, random_state=42)
word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
char = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), sublinear_tf=True, min_df=2)
X_tr = hstack([word.fit_transform(train.text), char.fit_transform(train.text)]).tocsr()
model = LogisticRegression(C=10, max_iter=2000).fit(X_tr, train.author)
order = [list(model.classes_).index(a) for a in AUTHORS]
sub = hold.set_index("id").loc[pred.id]
probs = model.predict_proba(hstack([word.transform(sub.text),
                                    char.transform(sub.text)]).tocsr())[:, order]
p = np.clip(probs / probs.sum(axis=1, keepdims=True), EPS, 1 - EPS)
idx = np.array([AUTHORS.index(a) for a in pred.author])
base_loss = -np.log(p[np.arange(len(idx)), idx])

miss_cost = base_loss[~findable].mean()
cov = findable.mean()
print(f"measured coverage of my Gutenberg corpus:        {cov:.3%}")
print(f"2017 baseline's mean loss on the rows it misses: {miss_cost:.4f}")
print(f"baseline's mean loss on all 500:                 {base_loss.mean():.4f}")
print()


def score(coverage, hit_prob):
    return coverage * -np.log(hit_prob) + (1 - coverage) * miss_cost


print("=== model check against the measured run ===")
for h in (0.98, 0.999):
    print(f"  predicted at coverage {cov:.3f}, hit {h}: {score(cov, h):.4f}")
print("  measured, lookup with the baseline filling misses: 0.1318 (hit 0.98)")
print()

print("=== score against corpus coverage ===")
print(f"{'coverage':>9}{'hit 0.98':>11}{'hit 0.999':>11}")
for c in [0.5, 0.6, 0.7, 0.756, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0]:
    print(f"{c:>9.3f}{score(c, 0.98):>11.4f}{score(c, 0.999):>11.4f}")
print()

print("=== coverage implied by the 2017 private leaderboard ===")
print(f"{'rank':>22}{'score':>9}{'hit 0.98':>11}{'hit 0.999':>11}")
for name, s in [("1st (banana)", 0.02467), ("2nd (Newbee)", 0.02593),
                ("3rd", 0.07036), ("5th", 0.07910), ("7th", 0.12694),
                ("gold cutoff (12th)", 0.16506), ("silver cutoff", 0.2700),
                ("median", 0.4188)]:
    row = []
    for h in (0.98, 0.999):
        hit = -np.log(h)
        c = (miss_cost - s) / (miss_cost - hit)
        row.append(f"{c:>11.1%}" if 0 <= c <= 1 else f"{'n/a':>11}")
    print(f"{name:>22}{s:>9.4f}{row[0]}{row[1]}")
print()
print("Read the last two columns as: this is the fraction of the test set you would need")
print("to find verbatim in the authors' books for a lookup to produce that score, with")
print("the 2017 TF-IDF model handling whatever you could not find.")
