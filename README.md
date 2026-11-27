# Guessing or Asking? LLM listeners at trouble points in clinical child conversation

Analysis code for the paper *Guessing or Asking? How LLM Listeners
Respond to Trouble in Clinical Child Conversation* (ACSW 2027, under review).

Five locally run LLMs (Gemma 4 26B-A4B, Qwen3.8 27B, Mistral Small 3.1 24B,
GPT OSS 20B, Llama 3.2 3B) took the adult's turn in 4,117 items from six clinical
CHILDES corpora, under three instructions (plain, child, repair)

## Quick start

```bash
pip install -r requirements.txt
python3 analysis/qwen_analysis.py          # Tables 4-5: per-cell SDT and differences
python3 analysis/qwen_error_turns.py       # error-turn table (RQ3)
python3 analysis/gemma_judge_analysis.py   # second-judge comparison (Table 6, appendix)
python3 analysis/self_preference_check.py  # blind human self-preference check
```

Run every script from the repository root with the coded data placed in
`data/` as shown below. Each script writes its table to `results/`, which it
creates on first run. Both folders are ignored by git.

## Layout

```
analysis/                 statistics behind every table and number in the paper
pipeline/                 Gemma 4 second judge, judge-free proxy, human-check pack
data/                     coded data (see "Data" below; ignored by git)
  qwen_judge/             primary judge: codes for the 6,000 sampled replies,
                          the sample manifest, gold validation, coverage per cell
  gemma_judge/            second judge: Gemma 4 codes and parse failures
  human_check/            150-reply blind human check: labels and reply sources
results/                  created by analysis/ (paper_*.csv, ignored by git)
```

## What each output feeds in the paper

| Output | Paper |
| --- | --- |
| `results/paper_cells.csv` | Table 4 (hits, false alarms, d′, c, corrected values) |
| `results/paper_differences.csv` | Table 5 (model minus adults, repair minus plain) |
| `results/paper_second_judge.csv` | Table 6 and the appendix comparison of the two judges |
| `results/paper_categories.csv` | reply categories (Figure 4 and the appendix table) |
| `results/paper_error_turns.csv` | Appendix table of replies at error turns |
| `results/paper_self_preference.csv` | Appendix table of the blind human check |

## Method settings

- **Turn types.** `unintelligible` turns give the hit rate, `quiet_baseline`
  turns (clear, with no trouble in the previous three child turns) give the
  false-alarm rate, and `error` turns give the correction measures.
- **Clarification** is any of `open_request`, `restricted_request` or
  `candidate_check` in the nine-category codebook.
- **Rates of 0 or 1** use the log-linear correction (k + 0.5)/(n + 1).
- **Intervals** come from 1,000 bootstrap resamples of children with seed 42,
  and differences between cells reuse the same resamples.
- **Corrected rates** use the Rogan–Gladen estimator with the judge's held-out
  sensitivity 0.593 and specificity 0.921. A cell whose observed rate is at or
  above 0.593, or at or below 0.079, reaches the bound of the correction and is
  marked † in the paper.


## Requirements

Python 3.10 or later with `numpy`.
