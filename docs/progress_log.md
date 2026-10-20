# Progress log

Capstone: replication of Moon et al. (2024), *Advancing ensemble learning techniques for residential building
electricity consumption forecasting: Insight from explainable artificial intelligence*, PLOS ONE 19(11): e0307654.

One entry per working session, newest last. Each entry: what was done, results, decisions, problems and how they were
solved, what is open. Details behind every point are in [replication_notes.md](replication_notes.md); results tables
are in [results/](results/).

---

## Overview (kept up to date)

| Part of the paper | Status | Match with the paper |
|---|---|---|
| Data, hourly table, 15 inputs (Tables 1–2) | Done | Exact: Table 1 for both datasets; our inputs equal the authors' published table on all 3,289 hours |
| Tree ensembles, dormitory (Tables 5–7, 11) | Done | LightGBM, CatBoost exact; GBM within 0.01; RF within 0.05; XGBoost within 0.08 |
| Tree ensembles, AEP (Tables 8–10, 12) | Done | RF, CatBoost exact; LightGBM-internal within 0.04; GBM, XGBoost within 1–3 points |
| Taylor diagrams (Figs 2–3) | Done | Point for point |
| SHAP / feature importance / PDP (Figs 4–11) | Done | Figs 4, 8, 10, 11 exact; 5, 9 near-exact; 6–7 close (paper inconsistent with its own Fig 4) |
| Raw-column experiment (Figs 12–13) | Done | RF and Fig 13 very close; paper's R² shown to be in-sample |
| Deep networks, dormitory (Table 13) | Done | Same conclusion (all worse than GBM); values mostly within 1 point |
| Deep networks, AEP (Table 14) | Done | CatBoost best on CVRMSE and NMAE, as in the paper; on MAPE six of the paper's networks beat CatBoost, none of ours do; overfit at 100 epochs; with the notebook early stopping CVRMSE matches, MAPE 15–25 points higher |
| Wrap-up (figures, README) | Done | – |
| Report | Not started | – |

---

## 2026-10-19: setup, data, tree ensembles

**Done**
- Read the article in full (including its peer-review history) and found the authors' repository
  (github.com/sodayeong/PLOS-ONE_Github): three notebooks and two CSVs, including the University Residential Complex
  (dormitory) feature table, which the article describes as not public. Both datasets are therefore in scope.
- Set up the project: public GitHub repository (cafeina13/moon2024-replication), Kaggle for all heavy computation
  (the laptop is only used for light work), data downloaded and stored inside Kaggle.
- Built the hourly AEP table and the article's 15 inputs from the raw UCI data, with automated tests.
- Ran the article's grid search with 5-fold cross-validation for 5 models × 3 input sets × 2 datasets
  (about 10,000 model fits for XGBoost alone), split over 13 Kaggle jobs.
- Wrote the comparison tables (ours vs the paper, cell by cell).

**Results**
- Table 1 reproduced exactly for both datasets.
- Our 15 inputs are identical to the authors' published feature table (3,289 hours × 17 columns).
- Random forest reproduces Tables 8, 9, 10 and 12 to every digit.

**Decisions**
- The article is the specification; where it is silent, the authors' code and data decide; where they contradict
  the article, the article is followed and the difference recorded.
- Ambiguous points settled from the authors' published table: hours are labelled by their end; weather is the
  reading at the hour mark; the day of an hour comes from its start time (00:00 belongs to the previous day);
  Sunday = 0; the holiday indicator marks weekends only; the first week is kept with zeros.

**Problems and solutions**

| Problem | Cause | Solution |
|---|---|---|
| pandas' default hourly grouping gives a maximum of 3,650 Wh instead of Table 1's 4,000 | Hours labelled by their start instead of their end | Label by the end; Table 1 now matches exactly (automated test) |
| `Cons_avg` defined only in words in the article | – | Tested candidate rules against the authors' table; "same hour, 7 most recent days of the same type" matches 3,121 / 3,121 hours |
| 137 mismatches in day of week, all at 00:00 | The hour labelled 00:00 belongs to the previous day | Day taken from the hour's start time |
| Brief claimed HM(41.35, 68.69, 42.61) = 48.23 | The paper computed HM from unrounded values (48.22 from rounded ones) | Test checks all 30 harmonic means of Tables 11–12 within 0.01 |
| Kaggle allows only 5 CPU / 2 GPU jobs at once | Platform limit | A local queue pushes jobs as slots free up |

---

## 2026-10-20 (night): which settings did the paper use? Figures

**Done**
- Compared grid-search results with the paper: dormitory close, AEP far off (up to 18 points).
- Trained *every* grid setting and compared each with the paper's rows to find which one the paper used.
- Ran the explainability figures (SHAP, feature importance, PDP; Figs 4–11), the Taylor diagrams (Figs 2–3) and the
  raw-column experiment (Figs 12–13); compared each figure with the published one.
- Started the deep-network experiments (Tables 13–14) on Kaggle GPU.
- Wrote walkthroughs 01–05 to be able to explain and defend every part of the code.

**Results**
- **Main finding:** the published tables do not come from the grid search the article describes. They come from one
  fixed setting per model (hard-coded in the authors' notebook), reused for all input sets. With those settings:
  CatBoost reproduces all 27 AEP values exactly; the dormitory tables match within 0.01–0.08.
- The authors' settings are mid-ranked on AEP under cross-validation and on the test set: not tuned on AEP, and not
  chosen by looking at the test set.
- Figure 11 is CatBoost on the *training* hours: reproduced exactly. The article's text says GBM on the test set.
- Figure 12's R² values (up to 0.96) are in-sample. On held-out weeks the same models score R² ≤ 0.05.

**Decisions**
- Main tables use the authors' fixed settings (they reproduce the published numbers); the grid-search results are
  reported as "what the described method gives".
- Figures follow the published figures where they differ from the authors' notebook (Figs 6a, 7a two-dimensional
  PDPs; Fig 11 CatBoost, importance order).

**Problems and solutions**

| Problem | Cause | Solution |
|---|---|---|
| AEP grid-search results 10–18 points better (MAPE) than the paper | The paper used fixed settings, not per-input-set tuning | Found by testing every setting against the published rows; tables rerun with the authors' settings |
| Same code, RF slightly different on Kaggle than on the laptop | scikit-learn 1.6.1 vs 1.9.1 | Pinned scikit-learn 1.9.1 (it reproduces the paper) |
| LightGBM Table 5 two points off | Cross-validation near-tie (10.11% vs 10.32%) flipped | The second-ranked setting reproduces all nine values exactly |
| XGBoost 1–3 points off | Suspected XGBoost 2.0 default change | Tested XGBoost 1.7.6: not closer, hypothesis ruled out; recorded as unexplained |
| SHAP job crashed at Fig 6a | Newer scikit-learn rejects integer columns (Holi) in PDPs | Convert to float for the PDP |
| Dormitory RF SHAP took 34 min | Time-budget rule picked a fixed 2,000-hour sample without checking it fits | Sample size now chosen to fit the budget |
| Our Fig 11 did not match | Article text wrong (says GBM, test set) | Drew it four ways; CatBoost on training hours matches exactly |
| Our Taylor diagrams showed 5 points | Paper shows 15 (5 models × 3 input sets) | Redrawn; now matches point for point |
| Our Fig 13 had NSM on top | Included NSM, rv1, rv2, which the paper's figure doesn't show | Reran with the figure's 27 inputs |

---

## 2026-10-20 (afternoon): raw-column results, deep networks

**Done**
- Reran the raw-column experiment with the 27 inputs of the published Figure 13; compared Figs 12–13.
- Collected the dormitory deep-network results (10 networks, about 1–3 hours each on GPU).
- Started the AEP deep-network run.
- Wrote walkthrough 06 (raw-column experiment).

**Results**
- Fig 13 now matches closely (same top 7 with two swaps, same bottom 3); RF in Fig 12 almost identical
  (in-sample R² 0.930 vs 0.964). Held-out R² of every model ≤ 0.05 (RF −1.99).
- Dormitory deep networks: all worse than GBM, as in the paper (MAPE 6.86–8.88 with 24th-hour scoring vs GBM 5.13).
  Values differ from Table 13 (paper 6.4–15.1); the authors' notebook scoring (all 24 forecast hours) lands closer to
  the published numbers, which suggests the table was scored that way, not "24th hour only" as the text says. To be
  checked on AEP.

**Problems and solutions**

| Problem | Cause | Solution |
|---|---|---|
| Deep networks much slower than estimated (about 75 s per epoch on the dormitory data) | Reproducible-GPU setting and data size | Accepted; both jobs fit within Kaggle's 12-hour limit |
| Bi-LSTM-TCN CVRMSE 31.6 (others about 10) | Predictions of up to 18,452 kWh around Lunar New Year 2018: the authors' networks use linear activations, so outputs are unbounded for unusual inputs | Not a pipeline bug; recorded as a robustness weakness of the architecture |

**Open**
- AEP deep networks (Table 14), then the scoring question (24th hour vs all horizons).
- Walkthrough 07 (deep networks).
- Weekly report for Monday.

---

## 2026-10-20 (evening): AEP deep networks, Tables 13–14

**Done**
- Collected the AEP deep-network results; wrote Tables 13 and 14 (ours scored two ways, next to the paper).
- Finished walkthrough 07 (deep networks).

**Results**
- AEP: all ten networks are worse than CatBoost on all three metrics. In the paper CatBoost wins on CVRMSE and NMAE
  only; six of its networks have a lower MAPE than CatBoost's 41.35. (Corrected 2026-10-21: first written as "the
  paper's conclusion holds".)
- Our AEP values are much worse than the paper's: MAPE 47–142 vs 34–50.
- The scoring question stays open. On AEP neither scoring is systematically closer to the paper.

**Problems and solutions**

| Problem | Cause | Solution |
|---|---|---|
| AEP networks predict −4,113 to 6,077 Wh (actual 10–1,100) | Overfitting: about 2,300 training windows, 100 epochs, no early stopping; linear activations leave outputs unbounded | Tested the authors' notebook early stopping (next entry) |

---

## 2026-10-20/21: early-stopping test, wrap-up

**Done**
- Added `run_deep --early-stopping` (the authors' notebook settings) and reran the ten AEP networks on Kaggle.
- Wrap-up: final figures in `docs/figures`, results summary and code map in the README.

**Results**
- With early stopping, nine networks stop after 11–18 epochs. The exploding predictions disappear, and CVRMSE
  (78–93) lands in the paper's range (81–98).
- MAPE improves for 8 of 10 networks (52–73) but stays 15–25 points above the paper's (34–50). It's driven by
  low-consumption hours, where small errors in Wh are large in percent.
- Every network is still worse than CatBoost on all three metrics.

**Decisions**
- The main Table 14 stays the article's 100-epoch run. The early-stopping run is reported next to it as a diagnosis
  (`docs/results/table14_early_stopping.md`).

**Open**
- Monday talking points; final report in Turkish.
