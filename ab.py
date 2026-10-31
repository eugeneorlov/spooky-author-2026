"""Did the harness injecting CLAUDE.md move the score? Same sentences, paired."""

import numpy as np
import pandas as pd

AUTHORS = ["EAP", "HPL", "MWS"]
EPS = 1e-15
import os
P = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)

clean = pd.read_csv(f"{P}/predictions_claude-opus-5-5_clean_500.csv")
dirty = pd.read_csv(f"{P}/predictions_claude-opus-5-5_dirty_120.csv")

common = dirty.id.tolist()
assert set(common).issubset(set(clean.id)), "samples do not overlap"
c = clean.set_index("id").loc[common]
d = dirty.set_index("id").loc[common]
assert (c.author.values == d.author.values).all()
print(f"paired on {len(common)} identical sentences\n")


def losses(frame):
    p = frame[AUTHORS].to_numpy(dtype=float)
    p = p / p.sum(axis=1, keepdims=True)
    p = np.clip(p, EPS, 1 - EPS)
    idx = np.array([AUTHORS.index(a) for a in frame.author])
    return -np.log(p[np.arange(len(idx)), idx]), (p.argmax(axis=1) == idx).astype(float)


lc, ac = losses(c)
ld, ad = losses(d)

print(f"{'prompt':<34}{'log loss':>10}{'accuracy':>10}")
print(f"{'clean (no CLAUDE.md)':<34}{lc.mean():>10.4f}{ac.mean():>10.4f}")
print(f"{'dirty (CLAUDE.md injected)':<34}{ld.mean():>10.4f}{ad.mean():>10.4f}")

diff = ld - lc
idx = rng.integers(0, len(diff), size=(20000, len(diff)))
lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
print(f"\npaired difference (dirty - clean): {diff.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
print(f"accuracy difference: {ad.mean() - ac.mean():+.4f} "
      f"({int(ad.sum())} vs {int(ac.sum())} correct of {len(common)})")
print(f"sentences where the top choice changed: {int((d[AUTHORS].to_numpy().argmax(axis=1) != c[AUTHORS].to_numpy().argmax(axis=1)).sum())}")
print(f"mean absolute change in the true-author probability: "
      f"{np.abs(np.exp(-ld) - np.exp(-lc)).mean():.4f}")
print(f"\nCI straddles zero: {lo < 0 < hi}")
