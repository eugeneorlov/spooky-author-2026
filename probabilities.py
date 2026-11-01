"""What do the model's self-reported probabilities actually look like?

The metric wants a distribution and a language model has none to give, so the three
numbers come back as text. This measures what that text looks like: how coarse the
numbers are, whether they already sum to one, how far the stated confidence sits from
the realised accuracy, and what the all-in rows would have cost had one been wrong.
"""

import os

import numpy as np
import pandas as pd

AUTHORS = ["EAP", "HPL", "MWS"]
EPS = 1e-15
P = os.path.dirname(os.path.abspath(__file__))

d = pd.read_csv(f"{P}/predictions_claude-opus-5-5_clean_500.csv")
M = d[AUTHORS].to_numpy(float)
truth = np.array([AUTHORS.index(a) for a in d.author])
top = M.max(axis=1)
correct = M.argmax(axis=1) == truth
p_true = M[np.arange(len(truth)), truth]

print("=== how coarse are the numbers ===")
print(f"distinct values the top number takes: {pd.Series(top.round(4)).nunique()} across 500 rows")
print(pd.Series(top.round(4)).value_counts().head(8).to_string())
print()
print(f"distinct values anywhere in the three slots: "
      f"{pd.Series(M.ravel().round(4)).nunique()}")
print()

sums = pd.Series(M.sum(axis=1).round(4))
print("=== did it obey the simplex without being asked ===")
print(f"rows already summing to exactly 1: {int((sums == 1.0).sum())} of {len(sums)}")
print()

print("=== stated confidence against realised accuracy ===")
print(f"mean stated confidence in the top author: {top.mean():.4f}")
print(f"actual accuracy:                          {correct.mean():.4f}")
print(f"so it is underconfident by:               {correct.mean() - top.mean():.4f}")
print()
for v in (1.00, 0.99, 0.98, 0.97, 0.96, 0.95):
    m = np.isclose(top, v)
    if m.sum():
        print(f"  stated exactly {v:.2f}: n={m.sum():>3}  correct {correct[m].mean():.3f}")
print()

print("=== the all-in rows ===")
allin = np.isclose(top, 1.0)
print(f"rows with 1.0 on one author and 0.0 on the others: {int(allin.sum())}")
print(f"correct on those rows: {int(correct[allin].sum())} of {int(allin.sum())}")
print(f"cost of a single wrong all-in row on the final score: {-np.log(EPS) / len(d):.4f}")
print(f"the run's actual score, for comparison:               0.1318")
print()

print("=== floor on the true author ===")
print(f"lowest probability it ever placed on the true author: {p_true.min():.4f}")
zero = (M == 0).any(axis=1)
on_truth = ((M == 0) & (np.eye(3)[truth] == 1)).any(axis=1)
print(f"rows with an exact 0 somewhere: {int(zero.sum())}, "
      f"of which on the true author: {int(on_truth.sum())}")
