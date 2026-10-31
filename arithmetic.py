"""Check every derived number the post states, and describe baseline calibration.

Nothing here needs an LLM call.
"""

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

EPS = 1e-15
AUTHORS = ["EAP", "HPL", "MWS"]

import os
P = os.path.dirname(os.path.abspath(__file__))

print("=== hard-label arithmetic ===")
cost = -np.log(EPS)
print(f"cost of one confidently wrong label: -ln(1e-15) = {cost:.4f}")
print(f"uniform 1/3 for every row:           -ln(1/3)  = {-np.log(1/3):.4f}")
print()

for name, target in [("median", 0.4188), ("bronze", 0.2938), ("silver", 0.2700), ("gold", 0.1651)]:
    err = target / cost
    print(f"{name:7} {target:.4f} -> max error rate {err*100:.3f}%  i.e. accuracy {100-err*100:.3f}%")
print()

for acc in (0.95, 0.90, 0.99):
    print(f"hard labels at {acc:.0%} accuracy -> log loss {(1-acc)*cost:.3f}")
print()

print("=== baseline calibration on the real holdout ===")
df = pd.read_csv(f"{P}/data/gh_train.csv")
train, hold = train_test_split(df, test_size=0.2, stratify=df.author, random_state=42)

word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
char = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), sublinear_tf=True, min_df=2)
X_tr = hstack([word.fit_transform(train.text), char.fit_transform(train.text)]).tocsr()
X_te = hstack([word.transform(hold.text), char.transform(hold.text)]).tocsr()
model = LogisticRegression(C=10, max_iter=2000).fit(X_tr, train.author)
order = [list(model.classes_).index(a) for a in AUTHORS]
p = model.predict_proba(X_te)[:, order]

top = p.max(axis=1)
truth_idx = np.array([AUTHORS.index(a) for a in hold.author])
p_true = p[np.arange(len(p)), truth_idx]
correct = p.argmax(axis=1) == truth_idx

print(f"n = {len(p)}")
print(f"log loss {-np.mean(np.log(np.clip(p_true, EPS, 1-EPS))):.4f}   accuracy {correct.mean():.4f}")
print(f"mean top probability           {top.mean():.4f}")
print(f"share of rows with top > 0.99  {(top > 0.99).mean():.4%}")
print(f"share of rows with top > 0.95  {(top > 0.95).mean():.4%}")
print(f"lowest probability it ever gave the true author: {p_true.min():.5f}")
print(f"worst single-row loss: {-np.log(p_true.min()):.3f}")
print()
print("reliability (bucket by confidence):")
for lo, hi in [(0.34, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.01)]:
    m = (top >= lo) & (top < hi)
    if m.sum():
        print(f"  top in [{lo:.2f},{hi:.2f}): n={m.sum():5d}  mean confidence {top[m].mean():.3f}  actual accuracy {correct[m].mean():.3f}")

print()
print("=== word stats ===")
print(f"mean words per sentence: {df.text.str.split().str.len().mean():.2f}")
print(f"sentences with 20+ words: {(df.text.str.split().str.len() >= 20).mean():.1%}")
