#!/bin/bash
# train.csv without Kaggle credentials. Two independent mirrors, cross-checked by
# verify_data.py: the HF copy has integer labels, the GitHub copy keeps the strings.
set -eu
D="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$D"
curl -fsSL -o "$D/gh_train.csv" \
  "https://raw.githubusercontent.com/ngbolin/SpookyAuthorPy/master/train.csv"
curl -fsSL -o "$D/hf_train.csv" \
  "https://huggingface.co/datasets/hkadxqq/spooky-author-identification/resolve/main/train.csv"
wc -l "$D"/*.csv
