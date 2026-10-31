"""Cross-check the two keyless copies of train.csv against each other.

The HF mirror has the right row count but integer labels. The GitHub copy (from the
repo of a 2017 participant) keeps the EAP/HPL/MWS strings. If the two agree row for
row on id and text, and the integer-to-string map is a clean bijection, the data is
the real competition file and the string labels can be trusted.
"""

import sys

import pandas as pd

import os
P = os.path.dirname(os.path.abspath(__file__))
HF = f"{P}/data/hf_train.csv"
GH = f"{P}/data/gh_train.csv"

hf = pd.read_csv(HF)
gh = pd.read_csv(GH)

print(f"HF copy:     {hf.shape[0]} rows, columns {list(hf.columns)}")
print(f"GitHub copy: {gh.shape[0]} rows, columns {list(gh.columns)}")
print()

print("GitHub label counts:")
print(gh.author.value_counts().sort_index().to_string())
print()
print("HF label counts:")
print(hf.author.value_counts().sort_index().to_string())
print()

# Align on id and compare.
m = hf.merge(gh, on="id", suffixes=("_hf", "_gh"))
print(f"ids present in both: {len(m)} of {len(hf)} / {len(gh)}")
print(f"duplicate ids: hf={hf.id.duplicated().sum()} gh={gh.id.duplicated().sum()}")

text_same = (m.text_hf == m.text_gh).sum()
print(f"identical text on matched ids: {text_same} of {len(m)}")
if text_same != len(m):
    bad = m[m.text_hf != m.text_gh].head(3)
    for _, r in bad.iterrows():
        print("  MISMATCH", r.id)
        print("   hf:", repr(r.text_hf[:120]))
        print("   gh:", repr(r.text_gh[:120]))
print()

# The integer -> string map must be one-to-one.
xt = pd.crosstab(m.author_hf, m.author_gh)
print("integer label (HF rows) vs string label (GitHub cols):")
print(xt.to_string())
print()

clean = (xt > 0).sum(axis=1).eq(1).all() and (xt > 0).sum(axis=0).eq(1).all()
print(f"bijection clean: {clean}")
if clean:
    mapping = {int(i): xt.columns[xt.loc[i].argmax()] for i in xt.index}
    print(f"derived mapping: {mapping}")

print()
print("mean words per sentence:", round(gh.text.str.split().str.len().mean(), 2))
print("total words:", int(gh.text.str.split().str.len().sum()))

if len(m) != 19579 or text_same != len(m) or not clean:
    print("\nVERIFICATION FAILED")
    sys.exit(1)
print("\nVERIFICATION PASSED: 19,579 rows, texts agree, labels map one-to-one.")
