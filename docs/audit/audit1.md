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

## Part 3: progress log, walkthroughs, talking points, "Beyond the paper", audit1.md

Auditor's scripts: scratchpad `audit/part3/`. Checked and correct: every number in `significance.md` and
`stopping_aep.md` (recomputed with independent code); the stopping rules and the Diebold-Mariano and bootstrap code;
every verdict number in parts 1-2 of this file. Walkthroughs are gitignored study notes; they were fixed the same way.

| # | Class | Finding | Verdict | Change |
|---|---|---|---|---|
| 1 | ERROR | Log: "No epoch reaches the paper's MAPE" | Confirmed: LSTM-TCN below 50.39 in 18 epochs (best 45.1) | "9 of 10" |
| 2 | ERROR | Every-horizon scoring "closer" still in walkthrough 07 and the log | Confirmed (part 2 #1 fixed the notes only) | Fixed |
| 3 | MISLABEL | Report: the paper "doesn't say when training stops" | Confirmed: it says 100 epochs | "doesn't mention early stopping or a validation split" |
| 4 | OVERSTATEMENT | Log: ensemble-vs-deep "significant" | Confirmed for AEP: p 0.095 / 0.044 / 0.016 | Per dataset and loss |
| 5 | ERROR | Bootstrap days for the dormitory shifted by an hour (its hours are labelled by their start) | Confirmed: 366 days instead of 365 | Fixed; intervals moved by at most 0.04 |
| 6 | MINOR | Test labelled as HLN but uses Bartlett Newey-West | Confirmed | Described accurately; HLN form shown in brackets |
| 7 | UNSUPPORTED | 8% simulation not reproducible | Confirmed | `size_check` in the code; the file prints its result (8.0%) |
| 8 | MINOR | "Bound" row is a bound for MAPE only | Confirmed | Reworded |
| 9 | MISLABEL | "Epoch 100 (article)" row is the 90%-data run | Confirmed | Relabelled; the article setup's 81.5 stated |
| 10 | MINOR | Duplicate paragraph | Confirmed | Removed |
| 11 | OVERSTATEMENT | "Best epoch is very early (1-8)"; "rough daily shape" | Confirmed: that is patience 10's pick; over 100 epochs the minima are later | Reworded |
| 12 | INCONSISTENCY | "about 0.006" in walkthrough 07 | Confirmed | 0.006-0.018 |
| 13 | ERROR | "200 Wh hour costs 50 points" in walkthrough 07 | Confirmed | 48% for a 210 Wh hour |
| 14 | OVERSTATEMENT | Early-stopping CVRMSE "matches" / "comes close" | Confirmed: lower for 8 of 10 | Fixed in all files |
| 15 | MINOR | Explosions "around Lunar New Year" | Confirmed: also 3-4 March (4 hours) | Added |
| 16 | MINOR | "24 to 47 hours after the latest input" | Confirmed | Fixed |
| 17 | UNSUPPORTED | "about 6 hours" SHAP in walkthrough 05 | Confirmed | Fixed |
| 18 | MINOR | Fig 11 order "almost exactly" | Confirmed: exact | Fixed |
| 19 | MINOR | Base value "about 590" | Confirmed: 593.4-593.9, GBM 563.8 | Fixed |
| 20 | INCONSISTENCY | Walkthrough 06 not updated for the R² finding; evidence not saved | Confirmed | Walkthrough fixed; `r2_definition.py` writes `docs/results/fig12_r2.md` |
| 21 | ERROR | Grid CatBoost "off by 1-5 points" | Confirmed: up to 7.30 | Fixed |
| 22 | OVERSTATEMENT | Grid RF "Matches" | Confirmed: up to 0.61 (scikit-learn 1.6.1) | Fixed |
| 23 | INCONSISTENCY | XGBoost version "likely" vs "ruled out" | Confirmed: 1.7.6 is closer only on AEP internal inputs | Stated |
| 24 | OVERSTATEMENT | Authors' settings "mid-table" everywhere | Confirmed: XGBoost up to 465 of 486 | "none near the top" |
| 25 | MINOR | "Every result table (5-14) reports three measures" | Confirmed | Fixed |
| 26 | UNSUPPORTED | Authors' reply to the reviewer quoted | Confirmed: reply not in the text | Cites the article's own sentence |
| 27 | INCONSISTENCY | Log overview "Report: Not started" | Confirmed | Updated |
| 28 | OVERSTATEMENT | Dormitory networks "mostly within 1 point" (log, talking points) | Confirmed: 16 of 30 | Fixed |
| 29 | OVERSTATEMENT | Talking points: "our networks much worse" | Confirmed | MAPE only; LSTM-TCN better |
| 30 | OVERSTATEMENT | Talking points: one setting, "tables match" | Confirmed | Where exact stated |
| 31 | ERROR | Talking points: automated tests for scaling and windows | Confirmed: tests cover split and lagged inputs only | Fixed |
| 32 | OVERSTATEMENT | Talking points: "R² ≈ 0.96" | Confirmed: 0.63-0.96 | Fixed, with the squared-correlation finding |
| 33 | MINOR | Talking points omit LightGBM | Confirmed | Added |
| 34 | MINOR | Quote "for each input set" not the article's; experiment "running" | Confirmed | Fixed |
| 35 | MINOR | "about 10,000 fits for XGBoost" | Confirmed: 7,290 per dataset | Fixed |
| 36 | MINOR | "about 75 s per epoch" | Confirmed: 10.9-103.4, mean 59.2 | Fixed |
| 37 | MINOR | "0.01-0.08"; MAPE driver shown for GRU only | Confirmed | Fixed |

Result: 37 real, none rejected. Eight were fixes from parts 1-2 not carried into every file; from now on a fix is
searched for across all documents before it is called done.
