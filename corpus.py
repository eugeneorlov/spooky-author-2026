"""Shared loader for the Gutenberg corpora of the three authors.

Strips the Project Gutenberg header and footer, normalises text for matching, and
splits into sentences. Used by lookup.py and fullcorpus.py.
"""

import glob
import os
import re

P = os.path.dirname(os.path.abspath(__file__))
GUT = os.path.join(P, "data", "gutenberg")
AUTHORS = ["EAP", "HPL", "MWS"]

START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.S | re.I)
END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG", re.I)
QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"',
          "—": " ", "–": " ", "…": " "}


def strip_boilerplate(text):
    m = START.search(text)
    if m:
        text = text[m.end():]
    m = END.search(text)
    if m:
        text = text[:m.start()]
    return text


def norm(s):
    """Lowercase, punctuation to spaces, whitespace collapsed. Word-for-word matching."""
    for a, b in QUOTES.items():
        s = s.replace(a, b)
    s = s.lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def load_raw(code):
    """Every file for one author, boilerplate removed, concatenated."""
    parts = []
    for path in sorted(glob.glob(os.path.join(GUT, code, "*.txt"))):
        with open(path, "rb") as f:
            raw = f.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        parts.append(strip_boilerplate(text))
    return parts, len(parts)


def haystack(code):
    """One normalised string per author, padded so substring hits land on word boundaries."""
    parts, n_files = load_raw(code)
    return " " + norm(" ".join(parts)) + " ", n_files


SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]*[A-Z])")


def sentences(code, min_words=4):
    """Corpus split into sentence-ish units. Crude next to CoreNLP, good enough to train on."""
    parts, n_files = load_raw(code)
    out = []
    for text in parts:
        text = re.sub(r"\s+", " ", text)
        for s in SENT_SPLIT.split(text):
            s = s.strip()
            if len(s.split()) >= min_words:
                out.append(s)
    return out, n_files
