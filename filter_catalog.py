"""Pick the three authors' own books out of the Project Gutenberg catalogue.

Keeps English text records where every credited author (ignoring editors, translators
and illustrators, which the catalogue marks in brackets) is the target author, so
anthologies by several hands do not pollute one author's corpus.
"""

import csv
import os
import sys

P = os.path.dirname(os.path.abspath(__file__))
CAT = os.path.join(P, "data", "gutenberg", "pg_catalog.csv")
TARGETS = {"EAP": "Poe, Edgar Allan", "HPL": "Lovecraft, H. P.", "MWS": "Shelley, Mary Wollstonecraft"}

rows = []
with open(CAT, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r.get("Type") != "Text" or r.get("Language") != "en":
            continue
        people = [a.strip() for a in (r.get("Authors") or "").split(";") if a.strip()]
        authors = [a for a in people if "[" not in a]
        if not authors:
            continue
        for code, name in TARGETS.items():
            if all(name in a for a in authors):
                rows.append((code, r["Text#"], r["Title"].replace("\r", " ").replace("\n", " ")))
                break

for code in TARGETS:
    n = sum(1 for c, _, _ in rows if c == code)
    print(f"# {code}: {n} books", file=sys.stderr)
for code, num, title in rows:
    print(f"{code}\t{num}\t{title[:70]}")
