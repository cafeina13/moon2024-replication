# Audit 2 (2026-10-21): six parts, one agent at a time, no hints

DRAFT while parts run; fixes are applied only after all six parts report. The agents were told not to read
`docs/audit/` and to judge the documents fresh.

## Part 1: report summary, Tables 5-12, tuning section; README results table; docs/results tables 05-12

Checked and correct: every cell of tables 05-12 (Markdown and HTML) against an independent recomputation; the
summary counts (79/135, 0.08; 55/135, 3.26); every per-model range in the README; the day groups (AEP H = weekends
only; dormitory H = 104 weekend days + 15 weekday holidays); the fixed settings against the authors' notebook; the
tuning-section claims (0.607; 11.84); Table 1; the citation.

| # | Class | Finding | Verdict | Change (pending) |
|---|---|---|---|---|
| 1 | INCONSISTENCY | Summary card and README: Fig 12 close "for RF only"; the report's own verdict says RF and LightGBM | Confirmed (verdict changed in audit 1, card and README not) | "RF and LightGBM" |
| 2 | OVERSTATEMENT | Card: "AEP: ours worse on MAPE" | Confirmed: 9 of 10 (LSTM-TCN 46.89 vs 50.39); all 10 with early stopping | Counts |
| 3 | INCONSISTENCY | Fig 13 worded "closely" / "Very close"; WeekStatus colour pattern reversed against the paper | Wording confirmed; colour reversal to be checked in part 3 | One wording |
| 4 | OVERSTATEMENT | "11 exact" without "as CatBoost on the training hours"; Taylor "match exactly" vs "point for point" | Confirmed | Qualified |
| 5 | UNSUPPORTED | "The article describes a grid search per model and input set" | Confirmed: per input set is the authors' notebook | Attributed |
| 6 | MINOR | Footer commit hash stale | Confirmed: a generated file can't hold its own commit | Generation date instead |

## Part 2: deep networks (report Tables 13-14, results tables, walkthrough 07, deep outputs)

Checked and correct: all 66 published deep values; every cell of tables 13, 14 and 14-early-stopping; every JSON
score against the prediction CSVs; the walkthrough's ranges, counts and code descriptions; the early-stopping and
per-epoch claims (including TCN restored to epoch 99).

| # | Class | Finding | Verdict | Change (pending) |
|---|---|---|---|---|
| 1 | OVERSTATEMENT | Card: "AEP: ours worse on MAPE" (same as part 1 #2); also worse on CVRMSE 8 / NMAE 9 of 10 | Confirmed | Counts per metric and run |
| 2 | ERROR | Walkthrough 07: Bi-LSTM-TCN's CVRMSE 31.6 explained by the high explosions only | Confirmed: also -14,652 kWh at 2018-03-04 05:00 (actual 912) and -95 at 04:00 | Added |
| 3 | OVERSTATEMENT | Walkthrough 07: "every horizon is easier" | Confirmed: dormitory 10 of 10; AEP 4-7 of 10 | Per dataset |
| 4 | INCONSISTENCY | TCN / Bi-LSTM example placed under early stopping but true only for the 100-epoch run | Confirmed | Run named |
| 5 | MINOR | Paper issue: "(Table 14 below)" | Confirmed: it is above, and shows no prediction ranges | "above", ranges given |
| 6 | MISLABEL | Best-ensemble row in tables 13/14 uses "ours / paper (difference)" under a three-value header | Confirmed | Labelled |
| 7 | MINOR | Notebook issues incomplete: bigru-tcn has the same single-state code; file name with spaces; shared optimizer; seeding once vs per model | Confirmed by reading the notebook and code (TensorFlow behaviour not run) | Listed |
| 8 | MINOR | Walkthrough snippet lacks the `--track` branch | Confirmed | Updated |
| 9 | MINOR | Bi-LSTM 100-epoch MAPE "81.2" | Confirmed: 81.1486 | 81.1 |
| 10 | MINOR | Snippet omits the decoder LeakyReLU | Confirmed | Added |

## Pending additions (found while explaining the figures, 2026-11-01; to be applied with the fixes and audited)

- Figure 13: WeekStatus colours reversed against the paper. The paper's figure uses weekend = 1 (as its Table 2);
  we followed its Figure 13 text (weekend = 0). Rerun with weekend = 1 (Kaggle).
- New paper issue: the text on Figure 6 says consumption "peaks during holidays"; both panels (theirs and ours) and
  the data (working days 1,291 kWh/h, days off 1,065) show days off lower it.
- Report, Figure 12 table: puts the paper's R² (a squared correlation) next to our usual R², which hides the match.
  Add our squared-correlation column (RF 0.963 vs 0.964, LightGBM 0.831 vs 0.831, GBM 0.540 vs 0.630); XGBoost and
  CatBoost need those libraries installed locally (`r2_definition.py`).
