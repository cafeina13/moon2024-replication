# Audit 1 (2026-10-21): independent agents, no hints, one part at a time

Brief: read-only; recompute every number from `outputs/`; check article claims against the article text and code
claims against the code; no target or limit on findings. The auditors were told nothing about the conversation or
earlier corrections. Each finding was verified by Claude before any change.

## Part 1: report summary and tables, README, docs/results, paper.py

Auditor's check scripts: scratchpad `audit/part1/` (not in the repository). Checked and correct: all 366 published
values in `paper.py`; all ensemble and deep tables against a recomputation from the saved predictions; the summary
counts; the per-model ranges; the article quotes.

| # | Class | Finding | Verdict | Change |
|---|---|---|---|---|
| 1 | OVERSTATEMENT | Dormitory deep networks "close" | Confirmed: LSTM 6.98 vs 15.06; Bi-LSTM-TCN CVRMSE 31.55 vs 9.23; the other eight within 1.79 | README, report card |
| 2 | OVERSTATEMENT | AEP networks "much worse than the paper's" | Confirmed: LSTM-TCN beats the paper on all three metrics; with early stopping CVRMSE lower than the paper's for 8 of 10 | Worse on MAPE only |
| 3 | UNSUPPORTED | Gains credited to early stopping alone | Confirmed: `--early-stopping` also holds out 10% of training windows; TCN never stopped yet improved. Later settled by the per-epoch run: with the same 90% of the data, 100 epochs give mean MAPE 115.9 (81.5 with all data), stopping 61.2, so the gain is from stopping | Both changes named |
| 4 | MINOR | "best weights restored" for all networks | **Rejected after the per-epoch run.** The early-stopping run's TCN (58.69) equals the per-epoch run's epoch 99, its best validation epoch, not epoch 100 (66.90): this tf-keras restores the best weights at the end of training too. The table header first changed under this finding has been corrected back | Header now states it |
| 5 | UNSUPPORTED | Causes of the deep-network differences stated as fact | Confirmed: none tested | "possible causes (untested)" |
| 6 | ERROR | "R² up to 0.96 on the training hours"; in-sample stated as fact | Confirmed. Followed up in part 2, finding 5 | Reworded; then settled (part 2) |
| 7 | OVERSTATEMENT | "Tables 10 and 12 show CatBoost" best | Confirmed: in Table 10 GBM leads MAPE, RF CVRMSE | Table 12 only |
| 8 | OVERSTATEMENT | "With those settings the tables reproduce" | Confirmed: AEP GBM, XGBoost, LightGBM up to 3.3 points | Exact for RF and CatBoost, others within 3.3 |
| 9 | MISLABEL | Settings "hard-coded in the authors' notebook", one per model | Confirmed: the explainability notebook (the ensemble notebook runs a grid search); CatBoost learning rate differs per dataset | "per model and dataset, in the explainability notebook" |
| 10 | CONFOUND | Grid column mixes tuning with scikit-learn 1.6.1 vs 1.9.1 | Confirmed: same RF setting, values up to 0.61 apart | Note under the tuning table |
| 11 | MINOR | 16 / 18 points not findable in the tuning table | Confirmed: weekday values; harmonic means up to 11.84 | Both stated |
| 12 | MISLABEL | "H = holidays and weekends" | Confirmed: AEP marks weekends only | "(AEP: weekends only)" |
| 13 | MINOR | "our inputs equal the authors' table" | Confirmed: AEP only | "our AEP inputs" |
| 14 | MINOR | CVRMSE "comes close" | Confirmed: lower for 8 of 10 | Reworded |

Result: 13 real, 1 rejected on later evidence. One fix first carried a new slip ("within about 1.7"; GRU is
1.79), caught by re-checking the fix.

## Part 2: figures, the article's unsupported statements, replication_notes.md

Auditor's scripts: scratchpad `audit/part2/`. Checked and correct: every paper figure against ours (verdicts for
Figs 2-4, 8-11, 13), every article quote, the paper-issue numbers, `paper.py`, most notes numbers.

| # | Class | Finding | Verdict | Change |
|---|---|---|---|---|
| 1 | ERROR | Every-horizon scoring "closer for most networks" (dormitory) | Confirmed: closer on 3 / 7 / 3 of 10 (MAPE / CVRMSE / NMAE) | Notes |
| 2 | ERROR | AEP actual values "10-1,100 Wh" | Confirmed: 210-3,200 (10-1,100 is the raw 10-minute range) | Notes, log, walkthrough, talking points |
| 3 | ERROR | SHAP on all rows "would take hours" / "6 h 37 min" | Confirmed: CatBoost estimate 36.6 min; RF's not saved | "above the 30-minute budget" |
| 4 | INCONSISTENCY | Figs 5a, 5d made with the old fixed 2,000-row rule; current code sizes the sample | Confirmed | Stated in the notes |
| 5 | MISLABEL | Fig 12 verdict "RF very close; others differ" | Confirmed by eye: LightGBM's scatter is almost identical to the paper's, yet R² 0.706 vs 0.831. Followed up: the paper's R² is the squared correlation of in-sample predictions (LightGBM 0.831 = 0.831, RF 0.963 vs 0.964) | Verdict, notes, report, README; `run_raw_figures` now records it |
| 6 | OVERSTATEMENT | In-sample stated as fact in the notes | Confirmed at the time; now shown (finding 5) | Stated with the evidence |
| 7 | OVERSTATEMENT | "The tables come from one fixed setting per model" | Confirmed: exact only for four model-dataset pairs | Notes say where it is exact |
| 8 | ERROR | Grid XGBoost/CatBoost "0.1-0.7 points better" | Confirmed: XGBoost -1.24 to +0.01, CatBoost -0.81 to +0.04 | Notes |
| 9 | INCONSISTENCY | XGBoost version "not yet tested" vs "ruled out" | Confirmed | Unexplained; 1.7.6 is further off |
| 10 | INCONSISTENCY | scikit-learn 1.6.1 vs 1.9.1 | Confirmed: 1.6.1 for the grid search only | Notes |
| 11 | MISLABEL | Conclusions "THI and WCT matter most"; Fig 9 text on weather | Confirmed: the Conclusions say more than the other weather inputs; the weather remark is in the Fig 8 text | Removed |
| 12 | OVERSTATEMENT | "Tables 8-10 show the opposite" | Confirmed: holds for Tables 5-7; AEP all inputs lowest in 38 of 45 | Counts stated |
| 13 | OVERSTATEMENT | 6b/7b "from a different run" | Confirmed: a different subset of hours fits as well | "run or subset" |
| 14 | UNSUPPORTED | Remaining deep gap "within what initialisation and versions can produce" | Confirmed: untested | Removed |
| 15 | OVERSTATEMENT | Training loss "about 0.006" | Confirmed: 0.0058-0.0179 | Range |
| 16 | OVERSTATEMENT | Dormitory networks "most within 1 point" | Confirmed: 16 of 30 cells | Counts |
| 17 | OVERSTATEMENT | Fig 5 "one or two neighbouring swaps" | Confirmed: XGBoost's Temp moves two places | Reworded |
| 18 | MISLABEL | Fig 6 "contour peak" | Confirmed: top contour labels | Reworded |
| 19 | MINOR | Stray "(34-50)" | Confirmed | Fixed |
| 20 | MINOR | "costs 50 MAPE points"; no 200 Wh test hour | Confirmed | "48% error for that hour" at 210 Wh |

Result: 20 real, none rejected. Not checkable by the auditor (no saved data): cross-validation rankings, RF's SHAP
time estimate, the dart timing probe.
