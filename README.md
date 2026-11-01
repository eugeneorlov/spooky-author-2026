# Spooky Author Identification, nine years later

Kaggle's [Spooky Author Identification](https://www.kaggle.com/c/spooky-author-identification) ran as a Halloween playground competition in 2017. You get one sentence and have to say whether Edgar Allan Poe, H. P. Lovecraft or Mary Shelley wrote it, scored by multiclass log loss. The winners were stacked ensembles of n-gram models, Naive Bayes, logistic regression and a few recurrent nets.

This repo runs the 2017 recipe and a zero-shot frontier LLM against each other on identical sentences, with the competition's own metric, and then works out where the LLM's score actually comes from. The short answer is retrieval, and the same retrieval was available in 2017: a plain string search over 62 Project Gutenberg books, with the 2017 model handling whatever the search misses, scores the same 0.1318 as the frontier model.

## Results

500 held-out sentences, `claude-opus-5-5`, zero-shot.

| Predictor | Sentences | Log loss | 95% interval | Accuracy |
| --- | --- | --- | --- | --- |
| Baseline, full holdout | 3,916 | 0.3652 | | 85.9% |
| Baseline, same sample | 500 | 0.4433 | [0.374, 0.520] | 82.2% |
| Verbatim lookup, baseline fills the misses | 500 | 0.1318 | | 95.8% |
| LLM zero-shot, its own probabilities | 500 | 0.1318 | [0.105, 0.162] | 96.6% |
| LLM zero-shot, hard label | 500 | 1.1743 | | 96.6% |

The 2017 private leaderboard, for scale: 1st 0.02467, 2nd 0.02593, 3rd 0.07036, gold cutoff 0.16506, silver 0.2700, bronze 0.2938, median 0.4188, 1,242 teams. So the zero-shot run sits inside the gold band even at the top of its interval, and an order of magnitude off the winner. Intervals are 20,000-sample bootstraps; the paired per-sentence difference against the baseline is 0.3116 [0.2436, 0.3844]. Of 500 calls, zero failed and zero came back unparseable.

Keeping only the model's top answer as a hard label, which is what a chat window gives you, costs ln(10^15) = 34.5 per mistake, so the same 96.6% accuracy scores 1.1743. That is worse than submitting one third for every author (1.0986).

## The lookup

All three authors died before 1940 and all three are on [Project Gutenberg](https://www.gutenberg.org/), which is where the competition data came from. So before crediting a model with anything, `lookup.py` does the dumbest possible 2017 solution: download 62 books, lowercase them, strip the punctuation, and search each test sentence as a substring.

- Found verbatim in exactly one author: 378 of 500, so 75.6%. Found in more than one: zero.
- Of those 378 matches, the matched author was the true label 378 times out of 378.
- Coverage by author: Poe 95.0%, Shelley 96.4%, Lovecraft 33.5%, because Gutenberg US holds only the Lovecraft stories whose copyright has cleared.

Bet 0.98 on the matched author where the search hits and let the 2017 logistic regression handle the rest, and that scores 0.1318 at 95.8% accuracy against the frontier model's 0.1318 at 96.6%. Precision was perfect, so coverage is the only limit, which gives a one-line model of score against coverage (`winner_math.py`) that reproduces the measured run at the coverage actually achieved.

| Fraction of the test set found verbatim | Log loss (0.98 on a hit) | Log loss (0.999 on a hit) |
| --- | --- | --- |
| 75.6% (62 books) | 0.1318 | 0.1173 |
| 90% | 0.0659 | 0.0486 |
| 95% | 0.0431 | 0.0248 |
| 99% | 0.0248 | 0.0058 |

Inverted against the 2017 leaderboard, first place implies 99.0% coverage at 0.98 per hit or 95.0% at 0.999, third place implies 85 to 89%, the gold cutoff 66 to 68%, and the median 12 to 13%. For comparison, the best openly documented result anywhere is about 0.237: 0.24074 for the 25th-place stacked ensemble in 2017, and 0.23693 for a 2026 autonomous agent running DeBERTa-v3 and stacking against the real private test set.

The competition's rules permitted external data explicitly, and the request not to use it is on the Data page rather than in the rules. `FINDINGS.md` quotes both, along with a contemporaneous complaint from the competition's own closing thread.

## More data does not help. Contaminated data does.

`fullcorpus.py` trains the same TF-IDF and logistic regression on those 62 books instead of `train.csv`.

| Training data for the 2017 recipe | 500 sentences | Full 3,916 holdout |
| --- | --- | --- |
| `train.csv` 80% split, 15,663 sentences | 0.4433 / 82.2% | 0.3652 / 85.9% |
| Books only, holdout text removed, 63,106 | 0.4930 / 81.6% | 0.4353 / 83.3% |
| Books plus train split, holdout removed, 78,769 | 0.4005 / 85.4% | 0.3617 / 86.4% |
| Books plus train split, holdout left in, 80,270 | 0.2314 / 92.0% | 0.2016 / 93.6% |

Four times the data, cleanly held out, moves the full holdout from 0.3652 to 0.3617. Leave the holdout sentences in the corpus, which is what naive corpus training does, and nine-year-old code reaches 0.2016 at 93.6%.

## The model's own route

Three measurements say the model is doing the same retrieval.

- **It names the source.** A specific work was named for 87.6% of sentences. Where it named one, accuracy was 99.3% and log loss 0.062. Where it answered `unknown`, accuracy fell to 77.4% and log loss rose to 0.625.
- **It can recite.** Given the first 60% of a sentence and asked to continue, it produced the next five words verbatim for 69.4% of the 284 sampled sentences of 20 words or more.
- **Its bookshelf is bigger than mine.** The recitation probe hits at 70.5% on sentences the string search finds and 66.7% on sentences it does not, the same rate, so the 122 misses are a gap in Gutenberg US and not in the model.

The errors are all short sentences: 100% accuracy at 20 words or more, 92.1% below that, all 17 mistakes in the short group, median wrong sentence 9 words against 22 for a right one, and 14 of the 17 came with `work: unknown`. That is not evidence of stylometry, though. 82.4% of the short sentences are findable by string search, and 15 of the 17 errors are on sentences the search got right, so where a `grep` works the model is worse than a `grep`. Separating style from fuzzy recall would need prose by these authors that is not online.

## The probabilities are written, not measured

Log loss wants a distribution and a language model has none to hand over, so the three numbers come back as text tokens: a self-report about the answer rather than a measurement of it. Token logprobs over the three labels would be the right instrument and are not available through the CLI backend. `probabilities.py` measures what the self-report looks like.

- The numbers are coarse. The top probability takes 28 distinct values across 500 rows; 0.98, 0.97, 0.96 and 0.95 account for 310 of them.
- All three summed to exactly 1 in 500 of 500 rows without being asked.
- Mean stated confidence is 0.9217 against realised accuracy of 0.9660, so it is underconfident by 4.4 points and losing score for it. Every row stating 0.95 or more was correct, 347 of 347.
- It went all-in (1.0 and two zeros) on 11 rows and was right on all 11. One wrong all-in row would have cost 0.0691 of the final 0.1318.
- The lowest probability it ever placed on the true author was 0.10, and its worst single row cost 2.30 against the baseline's worst at 6.88.

The TF-IDF baseline is the better calibrated of the two, putting more than 0.99 on a single author for 19.4% of the holdout and being right 99.5% of the time when it does. Training directly on log loss produces that; instruction tuning produces a number that reads like confidence.

## Method

Kaggle never published the test labels, so everything is scored on a stratified 80/20 holdout of the 19,579 labelled training rows (seed 42). The metric is implemented as the rules define it: rescale each row to sum to one, clip to `[1e-15, 1 - 1e-15]`, average the negative log of the true author's probability.

The baseline is the 2017 recipe without the stacking: word 1-2 grams plus `char_wb` 1-5 grams, TF-IDF with `sublinear_tf` and `min_df=2`, one logistic regression at `C=10`. It trains in about half a minute on a CPU.

The LLM is asked for a probability per author and, in the same answer, for the title of the work if it recognises it. Calls go through the Claude Code CLI in headless mode (`--backend cli`), which uses an existing subscription login, so no API key is needed; `--backend sdk` uses `ANTHROPIC_API_KEY` instead. Every raw reply is cached to JSONL keyed by model and prompt, so an interrupted run resumes without paying twice.

The CLI is a coding harness, so by default it injects the working directory's `CLAUDE.md` and other context into the prompt. The clean runs strip that with a scratch cwd and a redirected `HOME` and `CLAUDE_CONFIG_DIR`. Whether it mattered is measured rather than assumed: 120 sentences rerun with the context left in moved log loss by +0.0125, 95% interval [-0.027, +0.055], accuracy identical at 114 of 120, and only 2 top answers changed. See `ab.py` and `logs/ab.txt`.

Corpus sentences are split with a regex rather than a parser, so the corpus-trained variants carry a distribution shift the competition data does not have. That is the honest explanation for the books-only variant scoring worse than `train.csv` despite four times the data.

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

The run outputs are committed, so these rerun with no model calls at all:

```bash
.venv/bin/python -I analyse.py        # naming, memorisation, calibration
.venv/bin/python -I analyse2.py       # bootstraps, sentence-length split, every error
.venv/bin/python -I ab.py             # the injected-context control
.venv/bin/python -I arithmetic.py     # hard-label arithmetic, baseline calibration
.venv/bin/python -I probabilities.py  # what the self-reported numbers look like
```

The lookup leg needs the books, about 20 MB. These import `corpus.py`, so run them without `-I`:

```bash
PY=.venv/bin/python bash scripts/get_gutenberg.sh   # 62 books, keyless, ~2 minutes
.venv/bin/python lookup.py                          # the string search, writes lookup_500.csv
.venv/bin/python fullcorpus.py --cap 30000          # the three training variants
```

`lookup_500.csv` is committed, so the two analyses built on it need no download:

```bash
.venv/bin/python -I lookup_probe.py   # findability against the recitation probe
.venv/bin/python -I winner_math.py    # coverage implied by each leaderboard score
```

## Files

| Path | What it is |
| --- | --- |
| `spooky.py` | The experiment: baseline, zero-shot LLM, memorisation probe, both scorings |
| `verify_data.py` | Cross-checks the two `train.csv` mirrors on id and text before anything is trusted |
| `analyse.py`, `analyse2.py`, `ab.py`, `arithmetic.py`, `probabilities.py` | The analyses behind the LLM numbers |
| `corpus.py`, `filter_catalog.py` | Gutenberg catalogue filtering and the normalised corpus loader |
| `lookup.py`, `lookup_probe.py`, `fullcorpus.py`, `winner_math.py` | The lookup leg |
| `FINDINGS.md` | Every measured number with the script and log it came from |
| `predictions_*.csv`, `lookup_500.csv` | Per-sentence probabilities, named work, search hits, true author |
| `cache_*.jsonl` | Every raw model reply, so the run is auditable and the analysis is free |
| `logs/` | Console output of each run as it happened |

`data/` is not committed. The competition data belongs to Kaggle and the books belong to Project Gutenberg; the two scripts in `scripts/` fetch them.

## Caveats

The 0.1318 is a holdout split of the training file, not the 8,392 real test rows, which Kaggle never labelled publicly. Treat it as gold-medal territory on the same metric rather than as a leaderboard finish. Single model, single prompt, 500 sentences, one seed. The implied-coverage figures are an inference from a validated model of score against coverage, not a claim about what any particular 2017 team did.
