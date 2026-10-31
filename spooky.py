"""
Spooky Author Identification, nine years later.

Three predictors on the same held-out sentences:
  1. baseline   - TF-IDF (word + character n-grams) and logistic regression, the 2017 recipe
  2. llm-probs  - a frontier LLM, zero-shot, asked for a probability per author
  3. llm-label  - the same LLM answers, kept as a hard label (what a chat answer gives you)

Plus a memorisation probe: give the model the first part of a sentence and ask it to
continue it word for word. Style knowledge cannot do that; having read the book can.

Kaggle's test labels are not public, so everything is scored on a stratified holdout
of train.csv, with the metric exactly as the competition defined it:
multiclass log loss, rows rescaled to sum to one, probabilities clipped to [1e-15, 1 - 1e-15].

Two ways to reach the model. --backend sdk needs ANTHROPIC_API_KEY. --backend cli shells
out to the Claude Code CLI instead, which uses an existing Claude subscription login, so
no API key is needed. Every raw reply is cached to JSONL, so a stopped run resumes free.

Usage:
  pip install scikit-learn pandas numpy scipy anthropic
  export ANTHROPIC_API_KEY=...
  python spooky.py --data train.csv --n-llm 300
  python spooky.py --data train.csv --baseline-only
  python spooky.py --data train.csv --backend cli --claude-bin claude
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

AUTHORS = ["EAP", "HPL", "MWS"]
NAMES = {"EAP": "Edgar Allan Poe", "HPL": "H. P. Lovecraft", "MWS": "Mary Wollstonecraft Shelley"}
EPS = 1e-15


def kaggle_log_loss(y_true, probs):
    """The competition metric: rescale rows, clip, mean negative log of the true class."""
    p = np.asarray(probs, dtype=float)
    p = p / p.sum(axis=1, keepdims=True)
    p = np.clip(p, EPS, 1 - EPS)
    idx = np.array([AUTHORS.index(a) for a in y_true])
    return float(-np.mean(np.log(p[np.arange(len(idx)), idx])))


def accuracy(y_true, probs):
    pred = [AUTHORS[i] for i in np.argmax(probs, axis=1)]
    return float(np.mean([a == b for a, b in zip(y_true, pred)]))


# 1. The 2017 recipe -----------------------------------------------------------

def baseline(train, test, C=10):
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), sublinear_tf=True, min_df=2)
    X_train = hstack([word.fit_transform(train.text), char.fit_transform(train.text)]).tocsr()
    X_test = hstack([word.transform(test.text), char.transform(test.text)]).tocsr()
    model = LogisticRegression(C=C, max_iter=2000)
    model.fit(X_train, train.author)
    order = [list(model.classes_).index(a) for a in AUTHORS]
    return model.predict_proba(X_test)[:, order]


# 2 and 3. Zero-shot LLM -------------------------------------------------------

CLASSIFY_PROMPT = """Here is one sentence from a work of fiction. It was written by exactly one of:
EAP = Edgar Allan Poe
HPL = H. P. Lovecraft
MWS = Mary Wollstonecraft Shelley

Sentence: {text}

Give your probability for each author, and name the work if you recognise it.
Answer with JSON only, in this shape:
{{"EAP": 0.0, "HPL": 0.0, "MWS": 0.0, "work": "title or unknown"}}"""

CONTINUE_PROMPT = """Continue this sentence from a public-domain work of fiction, word for word as
the author wrote it. Give only the continuation, at least {k} words, nothing else.

{prefix}"""

SYSTEM_PROMPT = "Answer exactly what is asked, with no preamble and no explanation."


class Caller:
    """Sends one prompt at a time to the model and caches every raw reply to JSONL.

    Two backends: the Anthropic SDK (needs an API key) and the Claude Code CLI
    (uses an existing subscription login). Failures return "" and are counted by
    the caller, never silently turned into a guess.
    """

    def __init__(self, backend, model, cache_path, claude_bin="claude", retries=4,
                 cwd=None, env=None):
        self.backend = backend
        self.model = model
        self.claude_bin = claude_bin
        self.retries = retries
        self.cwd = cwd
        self.env = {**os.environ, **(env or {})} if env else None
        self.cache_path = cache_path
        self.cache = {}
        self.lock = threading.Lock()
        self.failures = 0
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    self.cache[r["key"]] = r["reply"]
            print(f"cache: loaded {len(self.cache)} replies from {cache_path}")
        if backend == "sdk":
            import anthropic
            self.client = anthropic.Anthropic(max_retries=6)

    def _key(self, prompt):
        return hashlib.sha256(f"{self.model}\x00{prompt}".encode()).hexdigest()[:32]

    def _store(self, key, prompt, reply):
        with self.lock:
            self.cache[key] = reply
            if self.cache_path:
                with open(self.cache_path, "a") as f:
                    f.write(json.dumps({"key": key, "model": self.model,
                                        "prompt": prompt, "reply": reply}) + "\n")

    def _sdk(self, prompt, max_tokens):
        msg = self.client.messages.create(
            model=self.model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    def _cli(self, prompt):
        # --tools "" leaves the model nothing to call, so the reply is a single turn.
        # --strict-mcp-config with no --mcp-config keeps 800 subprocesses from each
        # booting the user's MCP servers. Runs from a scratch cwd so no project
        # CLAUDE.md is picked up into the prompt.
        cmd = [self.claude_bin, "-p", "--model", self.model,
               "--system-prompt", SYSTEM_PROMPT, "--output-format", "text",
               "--strict-mcp-config", "--tools", ""]
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                           timeout=300, cwd=self.cwd, env=self.env)
        if r.returncode != 0:
            why = ((r.stderr or "") + " " + (r.stdout or "")).strip()[:200]
            raise RuntimeError(why or f"exit {r.returncode}")
        return r.stdout

    def __call__(self, prompt, max_tokens=200):
        key = self._key(prompt)
        if key in self.cache:
            return self.cache[key]
        last = ""
        for attempt in range(self.retries):
            try:
                reply = self._sdk(prompt, max_tokens) if self.backend == "sdk" else self._cli(prompt)
                if reply.strip():
                    self._store(key, prompt, reply)
                    return reply
                last = "empty reply"
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
        with self.lock:
            self.failures += 1
        print(f"  call failed after {self.retries} tries: {last}")
        return ""


def parse_probs(raw):
    """Pull the JSON object out of the reply. Unparseable replies get a uniform guess."""
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        d = json.loads(m.group(0))
        p = np.array([float(d.get(a, 0.0)) for a in AUTHORS])
        work = str(d.get("work", "unknown"))
    except (AttributeError, ValueError, TypeError):
        return np.full(3, 1 / 3), "unparsed"
    if not np.all(np.isfinite(p)) or p.sum() <= 0:
        return np.full(3, 1 / 3), work
    return p / p.sum(), work


def llm_classify(call, texts, workers=8):
    def one(t):
        return parse_probs(call(CLASSIFY_PROMPT.format(text=t)))

    with ThreadPoolExecutor(workers) as pool:
        out = list(pool.map(one, texts))
    probs = np.vstack([p for p, _ in out])
    works = [w for _, w in out]
    return probs, works


def words(s):
    return re.findall(r"[a-z']+", s.lower())


def memorisation_probe(call, texts, k=5, min_words=20, workers=8, limit=None):
    """Share of sentences where the model reproduces the next k words of the real text.

    Strict counts a match only when the reply opens with those k words. Lenient also
    counts them appearing anywhere in the first 25 words, which forgives a model that
    adds "Here is the continuation" despite being told not to.
    """
    long = [t for t in texts if len(t.split()) >= min_words]
    if limit:
        long = long[:limit]

    def one(t):
        tokens = t.split()
        cut = int(len(tokens) * 0.6)
        prefix, truth = " ".join(tokens[:cut]), words(" ".join(tokens[cut:]))[:k]
        reply = words(call(CONTINUE_PROMPT.format(k=k, prefix=prefix), 100))[:25]
        if len(truth) < k:
            return None
        strict = reply[:k] == truth
        lenient = any(reply[i:i + k] == truth for i in range(max(0, len(reply) - k + 1)))
        return strict, lenient

    with ThreadPoolExecutor(workers) as pool:
        out = [r for r in pool.map(one, long) if r is not None]
    if not out:
        return float("nan"), float("nan"), 0
    strict = float(np.mean([s for s, _ in out]))
    lenient = float(np.mean([l for _, l in out]))
    return strict, lenient, len(out)


# Run --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train.csv")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--n-llm", type=int, default=500, help="holdout sentences sent to the LLM")
    ap.add_argument("--n-probe", type=int, default=0, help="cap on memorisation probe calls, 0 = no cap")
    ap.add_argument("--no-probe", action="store_true", help="skip the memorisation probe")
    ap.add_argument("--sample-pool", type=int, default=0,
                    help="draw this many holdout sentences, then keep the first --n-llm of them; "
                         "lets a smaller run use a prefix of a bigger run's exact sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--backend", choices=["sdk", "cli"], default="sdk")
    ap.add_argument("--claude-bin", default="claude")
    ap.add_argument("--cli-cwd", default=None, help="scratch cwd for CLI calls, avoids CLAUDE.md pickup")
    ap.add_argument("--cli-env", action="append", default=[], metavar="KEY=VALUE",
                    help="env override for CLI calls, repeatable")
    ap.add_argument("--cache", default="llm_cache.jsonl")
    ap.add_argument("--out", default="llm_predictions.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    train, hold = train_test_split(df, test_size=0.2, stratify=df.author, random_state=args.seed)
    pool = max(args.sample_pool, args.n_llm)
    sample = hold.sample(n=min(pool, len(hold)), random_state=args.seed).head(args.n_llm)

    rows = []
    p_base_hold = baseline(train, hold)
    rows.append(("baseline, full holdout", len(hold),
                 kaggle_log_loss(hold.author, p_base_hold), accuracy(hold.author, p_base_hold)))
    p_base = p_base_hold[[hold.index.get_loc(i) for i in sample.index]]
    rows.append(("baseline, LLM sample", len(sample),
                 kaggle_log_loss(sample.author, p_base), accuracy(sample.author, p_base)))

    if not args.baseline_only:
        env = dict(kv.split("=", 1) for kv in args.cli_env)
        call = Caller(args.backend, args.model, args.cache, args.claude_bin,
                      cwd=args.cli_cwd, env=env)
        p_llm, works = llm_classify(call, sample.text.tolist(), args.workers)
        p_label = np.eye(3)[np.argmax(p_llm, axis=1)]
        rows.append(("LLM zero-shot, its probabilities", len(sample),
                     kaggle_log_loss(sample.author, p_llm), accuracy(sample.author, p_llm)))
        rows.append(("LLM zero-shot, hard label", len(sample),
                     kaggle_log_loss(sample.author, p_label), accuracy(sample.author, p_label)))

        named = np.array([w.strip().lower() not in ("unknown", "unparsed", "") for w in works])
        if args.no_probe:
            strict = lenient = float("nan")
            n_probe = 0
        else:
            strict, lenient, n_probe = memorisation_probe(
                call, sample.text.tolist(), workers=args.workers, limit=args.n_probe or None)

        pd.DataFrame({"id": sample.id, "author": sample.author, "text": sample.text, "work": works,
                      **{a: p_llm[:, i] for i, a in enumerate(AUTHORS)}}
                     ).to_csv(args.out, index=False)

    print(f"\n{'predictor':<34}{'n':>6}{'log loss':>10}{'accuracy':>10}")
    for name, n, ll, acc in rows:
        print(f"{name:<34}{n:>6}{ll:>10.3f}{acc:>10.3f}")
    print("\n2017 private leaderboard for scale: 1st 0.0247, gold 0.165, silver 0.270, "
          "bronze 0.294, median 0.419")

    if not args.baseline_only:
        n_bad = sum(w == "unparsed" for w in works)
        print(f"\nModel: {args.model} via {args.backend}")
        print(f"Hard failures (no reply, scored as 1/3 each): {call.failures}")
        print(f"Unparseable replies (scored as 1/3 each): {n_bad} of {len(works)}")
        if named.any():
            print(f"LLM named the work for {named.mean():.0%} of sentences "
                  f"(accuracy when named: {accuracy(sample.author[named], p_llm[named]):.3f}, "
                  f"when not: {accuracy(sample.author[~named], p_llm[~named]):.3f})")
        print(f"Memorisation probe on {n_probe} sentences with 20+ words: "
              f"next 5 words exact at the start {strict:.0%}, anywhere in the opening {lenient:.0%}")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
