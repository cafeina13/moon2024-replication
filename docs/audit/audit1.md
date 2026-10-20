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
| 3 | UNSUPPORTED | Gains credited to early stopping alone | Confirmed: `--early-stopping` also holds out 10% of training windows; TCN never stopped yet improved | Both changes named; overfitting called likely, not shown |
| 4 | MINOR | "best weights restored" for all networks | Confirmed: TCN ran 100 epochs; restoring happens only on an early stop | Table header |
| 5 | UNSUPPORTED | Causes of the deep-network differences stated as fact | Confirmed: none tested | "possible causes (untested)" |
| 6 | ERROR | "R² up to 0.96 on the training hours"; in-sample stated as fact | Confirmed: ours 0.31-0.98; the article doesn't say how R² was computed | "very probably in-sample", with the evidence |
| 7 | OVERSTATEMENT | "Tables 10 and 12 show CatBoost" best | Confirmed: in Table 10 GBM leads MAPE, RF CVRMSE | Table 12 only |
| 8 | OVERSTATEMENT | "With those settings the tables reproduce" | Confirmed: AEP GBM, XGBoost, LightGBM up to 3.3 points | Exact for RF and CatBoost, others within 3.3 |
| 9 | MISLABEL | Settings "hard-coded in the authors' notebook", one per model | Confirmed: the explainability notebook (the ensemble notebook runs a grid search); CatBoost learning rate differs per dataset | "per model and dataset, in the explainability notebook" |
| 10 | CONFOUND | Grid column mixes tuning with scikit-learn 1.6.1 vs 1.9.1 | Confirmed: same RF setting, values up to 0.61 apart | Note under the tuning table |
| 11 | MINOR | 16 / 18 points not findable in the tuning table | Confirmed: weekday values; harmonic means up to 11.84 | Both stated |
| 12 | MISLABEL | "H = holidays and weekends" | Confirmed: AEP marks weekends only | "(AEP: weekends only)" |
| 13 | MINOR | "our inputs equal the authors' table" | Confirmed: AEP only | "our AEP inputs" |
| 14 | MINOR | CVRMSE "comes close" | Confirmed: lower for 8 of 10 | Reworded |

Result: 14 real, none rejected. One fix first carried a new slip ("within about 1.7"; GRU is 1.79), caught by
re-checking the fix.
