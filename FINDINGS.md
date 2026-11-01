# Spooky Author, nine years later: all measured results

Every number here comes from a script in this folder and a log in `logs/`. Scored with the
competition metric on a stratified 80/20 holdout of `train.csv` (seed 42), because Kaggle
never released the test labels.

## 1. The three-way comparison, same 500 held-out sentences

| Predictor | Log loss | 95% interval | Accuracy | Source |
| --- | --- | --- | --- | --- |
| 2017 recipe, trained on `train.csv` | 0.4433 | [0.374, 0.520] | 82.2% | `logs/analysis2.txt` |
| 2017 recipe, + all 62 books, holdout text removed | 0.4005 | | 85.4% | `logs/fullcorpus.txt` |
| Verbatim lookup only, thirds where it misses | 0.2683 | | 77.6% | `logs/lookup.txt` |
| 2017 recipe, + all 62 books, holdout text left in | 0.2314 | | 92.0% | `logs/fullcorpus.txt` |
| Verbatim lookup, 2017 recipe fills the misses | 0.1318 | | 95.8% | `logs/lookup.txt` |
| `claude-opus-5-5` zero-shot, its own probabilities | 0.1318 | [0.105, 0.162] | 96.6% | `logs/analysis2.txt` |
| `claude-opus-5-5` zero-shot, hard label | 1.1743 | | 96.6% | `logs/analysis2.txt` |

On the full 3,916-sentence holdout: 2017 recipe 0.3652 at 85.9%, same recipe with the books
added and holdout text removed 0.3617 at 86.4%, same with holdout text left in 0.2016 at 93.6%.

2017 private leaderboard: 1st 0.02467, 2nd 0.02593, 3rd 0.07036, gold cutoff 0.16506,
silver 0.2700, bronze 0.2938, median 0.4188, 1,242 teams.

## 2. The lookup

62 books from Project Gutenberg: 21 Poe, 20 Lovecraft, 21 Shelley. Text normalised to
lowercase word sequences, punctuation stripped, then each holdout sentence searched in each
author's books.

- Found verbatim in exactly one author: 378 of 500 (75.6%). Found in more than one: 0.
- Of those 378, the match agreed with the label 378 times. Precision 100%.
- Coverage by author: Poe 95.0%, Shelley 96.4%, Lovecraft 33.5% (Gutenberg US holds only
  the Lovecraft stories whose copyright has cleared).
- By length: 82.4% of sentences under 20 words, 70.4% of sentences at 20 words or more.

Score against coverage, validated against the measured run at 75.6%:

| Corpus coverage | Log loss (0.98 on a hit) | Log loss (0.999 on a hit) |
| --- | --- | --- |
| 75.6% (measured) | 0.1318 | 0.1173 |
| 90% | 0.0659 | 0.0486 |
| 95% | 0.0431 | 0.0248 |
| 99% | 0.0248 | 0.0058 |

Reading the leaderboard through that model, a lookup would need 99.0% coverage to produce
first place at 0.98 confidence per hit, or 95.0% at 0.999. Third place implies 85 to 89%,
the gold cutoff 66 to 68%, the median 12 to 13%.

## 3. Was the lookup allowed in 2017

- The Specific Competition Rules contain an External Data Use clause that supersedes general
  rule 7.B: "You may use data, other than the Competition Data, as allowed on the Competition
  Website to develop and test your models and Submissions", subject to having the right to
  use it. Public-domain books qualify.
- The entry rules ban something narrower: "Submissions may not use or incorporate information
  from hand labeling or human prediction of the validation dataset or test data records."
  A programmatic join against the books is not hand labelling.
- The Data page asks, in the paragraph explaining that playground competitions are for fun,
  "In short, please don't look up the answers."
- In Kaggle's own closing thread, a participant placed 446th wrote: "Right now it seems that
  some users are for some reason trying to find the public answers on the web and submit those
  as predictions, and this makes the leaderboard less interesting." Same comment notes the
  absence of high-ranked Kagglers at the top of the leaderboard.
- Prizes were $2,000 per upvoted or themed kernel and $1,000 per top discussion topic. Nothing
  was awarded for leaderboard position, and the top two teams published no writeup.
- Best documented honest results anywhere: Stanwar 25th at 0.24074 (2017 stacked ensemble),
  a 2026 autonomous agent at 0.23693 (DeBERTa-v3 plus stacking, MLE-bench, real private test
  set), a 2026 classical rerun at 0.30414, a 2026 Qwen3.5-2B fine-tune at 0.3655.

## 4. Why the LLM scores what it does

- Named a specific work for 87.6% of sentences. Named: 99.3% accuracy, 0.062 log loss.
  Answered `unknown`: 77.4% accuracy, 0.625 log loss.
- Memorisation probe (first 60% of a sentence, continue it): next five words exact for
  69.4% of the 284 sentences at 20 words or more.
- The probe hits at 70.5% on sentences my corpus contains and 66.7% on sentences it does
  not, so the 122 misses are a gap in Gutenberg US rather than in the model. Lovecraft makes
  the point alone: 33.5% findable, 69.4% recited, 97.5% accuracy.
- All 17 errors are sentences under 20 words. 15 of the 17 are sentences a text search
  identified correctly. 14 of the 17 came with `work: unknown`.
- 100% accuracy at 20 words or more, 92.1% below. The 2017 recipe on the same split:
  88.7% and 73.6%.

## 5. Calibration, against expectation

- The model hedges rather than overclaims. Confidence 0.95 bucket: 249 rows, 100% correct.
  Confidence 0.98 bucket: 167 rows, 100% correct. It never put under 0.01 on the true author.
  Worst single row 2.30.
- The 2017 baseline is the calibrated one: more than 0.99 on one author for 19.4% of the
  holdout, right 99.5% of the time there, reliability within two points across the range.
  Its worst single row is 6.88.
- Collapsing to a hard label costs ln(10^15) = 34.5 per error, so 96.6% accuracy scores
  1.1743, worse than uniform thirds at 1.0986.

## 6. Method notes

- Calls went through the Claude Code CLI in headless mode, so no API key. 784 calls, twelve
  workers, 23 minutes, zero failures, zero unparseable.
- The CLI injects the working directory's `CLAUDE.md` into prompts. Stripped with a scratch
  cwd and a redirected `HOME` and `CLAUDE_CONFIG_DIR`, then measured: 120 sentences rerun with
  the context left in moved log loss by +0.0125, interval [-0.027, +0.055], accuracy identical
  at 114 of 120, two top answers changed.
- Corpus sentence splitting is a regex, not CoreNLP, so the corpus-trained variants carry a
  distribution shift the original data does not have. That is the honest explanation for
  variant A scoring worse than `train.csv` despite four times the data.

## 7. What the self-reported probabilities look like

`probabilities.py`, logged in `logs/probabilities.txt`. The metric wants a distribution and
the model has none to hand over, so the three numbers arrive as text tokens: a self-report
about the answer rather than a measurement of it. Token logprobs over the three labels would
be the proper instrument and were not available through the CLI backend.

- The numbers are coarse. The top probability takes 28 distinct values across 500 rows, and
  0.98, 0.97, 0.96 and 0.95 account for 310 of them. 51 distinct values appear anywhere in
  the three slots.
- All three summed to exactly 1 in 500 of 500 rows without being asked, so the metric's
  rescaling step was a no-op.
- Mean stated confidence in the top author is 0.9217 against realised accuracy of 0.9660,
  so it is underconfident by 4.4 points and leaving score on the table.
- Every row stating 0.95 or more was correct: 347 of 347.
- It went all-in (1.0 on one author, 0.0 on the others) on 11 rows and was right on all 11.
  One wrong all-in row would have cost 0.0691 on the final score, against a run score of
  0.1318. All 11 exact zeros fell on wrong authors.
- The lowest probability it ever placed on the true author was 0.10.
