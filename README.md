# Spooky Author Identification, nine years later

Kaggle's [Spooky Author Identification](https://www.kaggle.com/c/spooky-author-identification) ran as a Halloween playground competition in 2017. You get one sentence and have to say whether Edgar Allan Poe, H. P. Lovecraft or Mary Shelley wrote it, scored by multiclass log loss. The winners were stacked ensembles of n-gram models, Naive Bayes, logistic regression and a few recurrent nets.

This repo runs the 2017 recipe and a zero-shot frontier LLM against each other on identical sentences, with the competition's own metric, and then checks where the LLM's score comes from.

## Results

500 held-out sentences, `claude-opus-5-5`, zero-shot.

| Predictor | Sentences | Log loss | 95% interval | Accuracy |
| --- | --- | --- | --- | --- |
| Baseline, full holdout | 3,916 | 0.365 | | 85.9% |
| Baseline, same sample | 500 | 0.443 | [0.374, 0.520] | 82.2% |
| LLM zero-shot, its own probabilities | 500 | 0.132 | [0.105, 0.162] | 96.6% |
| LLM zero-shot, hard label | 500 | 1.174 | | 96.6% |

The 2017 private leaderboard, for scale: first place 0.0247, gold cutoff 0.165, silver 0.270, bronze 0.294, median 0.419. So the zero-shot run sits inside the gold band even at the top of its interval, and an order of magnitude off the winner. Intervals are 20,000-sample bootstraps; the paired per-sentence difference against the baseline is 0.312 [0.244, 0.384]. Of 500 calls, zero failed and zero came back unparseable.

Keeping only the model's top answer as a hard label, which is what a chat window gives you, costs ln(10^15) = 34.5 per mistake, so the same 96.6% accuracy scores 1.174. That is worse than submitting one third for every author (1.099).

## Where the score comes from

The model has read the books. All three authors are public domain on Project Gutenberg, and three measurements point the same way.

- **It names the source.** A specific work was named for 87.6% of sentences. Where it named one, accuracy was 99.3% and log loss 0.062. Where it answered `unknown`, accuracy fell to 77.4% and log loss rose to 0.625.
- **It can recite.** Given the first 60% of a sentence and asked to continue, it produced the next five words verbatim for 69.4% of the 284 sampled sentences of 20 words or more.
- **The errors are all short sentences.** 100% accuracy on sentences of 20 words or more, 92.1% below that, and all 17 mistakes in the short group. Median wrong sentence 9 words, median right one 22. Fourteen of the 17 came with `work: unknown`.

The 2017 rules asked participants not to look up the answers. Pretraining is the lookup.

On calibration, the result runs against the usual expectation. The model hedges rather than overclaims: 100% actual accuracy in both its 0.95 and 0.98 confidence buckets, never below 0.01 on the true author, worst single row 2.30. The TF-IDF baseline is the well-calibrated one, putting more than 0.99 on a single author for 19.4% of the holdout and being right 99.5% of the time when it does.

## Method

Kaggle never published the test labels, so everything is scored on a stratified 80/20 holdout of the 19,579 labelled training rows (seed 42). The metric is implemented as the rules define it: rescale each row to sum to one, clip to `[1e-15, 1 - 1e-15]`, average the negative log of the true author's probability.

The baseline is the 2017 recipe without the stacking: word 1-2 grams plus `char_wb` 1-5 grams, TF-IDF with `sublinear_tf` and `min_df=2`, one logistic regression at `C=10`. It trains in about 15 seconds on a CPU.

The LLM is asked for a probability per author and, in the same answer, for the title of the work if it recognises it. Calls go through the Claude Code CLI in headless mode (`--backend cli`), which uses an existing subscription login, so no API key is needed; `--backend sdk` uses `ANTHROPIC_API_KEY` instead. Every raw reply is cached to JSONL keyed by model and prompt, so an interrupted run resumes without paying twice.

The CLI is a coding harness, so by default it injects the working directory's `CLAUDE.md` and other context into the prompt. The clean runs strip that with a scratch cwd and a redirected `HOME` and `CLAUDE_CONFIG_DIR`. Whether it mattered is measured rather than assumed: 120 sentences rerun with the context left in moved log loss by +0.0125, 95% interval [-0.027, +0.055], accuracy identical at 114 of 120, and only 2 top answers changed. See `ab.py` and `logs/ab.txt`.

## Reproducing

```bash
uv venv && uv pip install scikit-learn pandas numpy scipy
bash scripts/get_data.sh                 # two keyless mirrors of train.csv
.venv/bin/python -I verify_data.py       # cross-check them against each other
.venv/bin/python -I spooky.py --data data/gh_train.csv --baseline-only
```

The LLM leg needs either an `ANTHROPIC_API_KEY` or a logged-in Claude Code CLI:

```bash
PY=.venv/bin/python CLAUDE_BIN=/path/to/claude bash scripts/run_llm.sh claude-opus-5-5 clean 500
```

The run outputs are committed, so the analysis reruns with no model calls at all:

```bash
.venv/bin/python -I analyse.py      # naming, memorisation, calibration
.venv/bin/python -I analyse2.py     # bootstraps, sentence-length split, every error
.venv/bin/python -I ab.py           # the injected-context control
.venv/bin/python -I arithmetic.py   # hard-label arithmetic, baseline calibration
```

## Files

| Path | What it is |
| --- | --- |
| `spooky.py` | The experiment: baseline, zero-shot LLM, memorisation probe, both scorings |
| `verify_data.py` | Cross-checks the two `train.csv` mirrors on id and text before anything is trusted |
| `analyse.py`, `analyse2.py`, `ab.py`, `arithmetic.py` | The analyses behind every number above |
| `predictions_*.csv` | Per-sentence probabilities, named work, and the true author |
| `cache_*.jsonl` | Every raw model reply, so the run is auditable and the analysis is free |
| `logs/` | Console output of each run as it happened |

`data/` is not committed. The competition data belongs to Kaggle; `scripts/get_data.sh` fetches it.

## Caveats

The 0.132 is a holdout split of the training file, not the 8,392 real test rows, which Kaggle never labelled publicly. Treat it as gold-medal territory on the same metric rather than as a leaderboard finish. Single model, single prompt, 500 sentences, one seed.
