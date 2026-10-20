# Replication notes

Choices made where Moon et al. (2024) is silent or inconsistent, one line per point.
Sources: the article ([doi:10.1371/journal.pone.0307654](https://doi.org/10.1371/journal.pone.0307654)) and
the authors' repository ([sodayeong/PLOS-ONE_Github](https://github.com/sodayeong/PLOS-ONE_Github), created
27 October 2024). Rule: the article decides; where it is silent, the authors' code and data decide; where the
code contradicts the article, we follow the article and report the code's variant.

## Data and features (AEP)

All points below are verified by `tests/test_data.py::test_features_match_authors_table`: our table, built from the
raw UCI data, equals the authors' published table on all 3,289 hours and 17 columns.

| Point | Article | What we do | Evidence |
|---|---|---|---|
| Hourly grouping | "resampled to hourly" | Sum of the six readings in (t - 1 h, t], labelled t; incomplete hours dropped | Reproduces Table 1 exactly |
| Hourly weather | Not stated | Reading at the hour mark t | Authors' table |
| Hour of day | Eqs (1)-(2) | Hour of the label t | Authors' table |
| Day of week | Eqs (3)-(4), numbering not stated | Sunday = 0 ... Saturday = 6, taken from the hour's start (00:00 belongs to the previous day) | Authors' table |
| Holiday indicator | Table 2: weekend = 1; text: public holidays added from timeanddate.com | Weekend only, from the hour's start; no public holidays | Authors' table marks none (Ascension, 5 May 2016, is 0). Article and data disagree; we follow the data that produced the results |
| First week | Not stated | Kept; Cons_7, Holi_7 and Cons_avg are 0 there | Authors' table |
| Cons_avg | "Average over the previous seven days" of the same day type | Mean at the same hour over the 7 most recent earlier days with the same Holi (fewer when history runs out) | Authors' table; matches 3,121 of 3,121 hours |
| Wind speed in WCT | Eq (6), unit not stated | m/s as given | Authors' table |
| Train/test split | Train to 16 April, test from 17 April 2016 | Train ends with the hour labelled 2016-04-17 00:00 (23:00-24:00 on 16 April): 2,311 / 978 hours | Authors' code (`iloc[:2311]`) |

## Figures 4-11 (explainability)

The published figures were compared with the authors' notebook `explainable_AI_energy_consumption_models.ipynb`;
where they differ, we follow the published figures.

| Figure | Authors' notebook | Published figure | What we do |
|---|---|---|---|
| 6a, 7a | One-dimensional PDP of Holi / THI | Two-dimensional contour PDP of Holi × Cons_avg / THI × Cons_avg | Two-dimensional |
| 11 | CatBoost, first 100 test hours, inputs ordered by hierarchical clustering | The text says GBM on the test set. The figure is **CatBoost on all 2,311 training hours**, importance order: our run with these settings reproduces it exactly (same order of all 15 inputs, same axis range, same lines) | CatBoost, training hours, importance order |
| All | SHAP values on the training part | Not visible (Figure 11: training part) | Training part. Figures 5a (RF) and 5d (CatBoost) use a random 2,000 of the 20,472 dormitory training hours, because exact SHAP on all of them was estimated above the 30-minute budget (CatBoost: 36.6 min; RF's estimate was not saved). These two panels come from the earlier fixed 2,000-row rule; the current `run_xai` sizes the sample to fit the budget, so a rerun would use more rows |

### How close the figures are

| Figure | Match | Notes |
|---|---|---|
| 2, 3 (Taylor) | Point for point | Each shows 15 models (5 ensembles × 3 input sets). The text on Figure 3 says correlation "about 0.6"; the diagram itself shows 0.3-0.5 (ours: 0.28-0.51) |
| 4, 8, 10, 11 | Exact | Same bars, same order, same individual dots and lines. 10a plots Hour_y's SHAP values coloured by Hour_x |
| 5 | LightGBM, CatBoost exact order; RF one neighbouring swap; XGBoost one swap and Temp two places higher | RF and CatBoost use a 2,000-hour sample; the XGBoost difference is unexplained (XGBoost 1.7.6 was tried and is further off) |
| 9 | RF exact; XGBoost, LightGBM nearly; GBM somewhat different | Same models that keep differences of up to 2.7 (GBM) and 3.3 (XGBoost) points in Tables 8-10 |
| 6, 7 | Close, not exact | Contour levels within about 21 kWh (top labels 412.24 vs 409.04). The paper's 6b/7b lack the few extreme SHAP values (Holi up to +135, THI up to +290) that its own Figure 4b, from the same GBM, shows; so 6b/7b come from a different run or a different subset of hours than 4b. Drawing them on the test year matches worse (PDP 511 vs 412, colour range 950-1,950 vs 800-1,700) |
| 13 | Very close | With the published figure's 27 inputs (no NSM, rv1, rv2): same top 7 (two neighbouring swaps), same bottom 3; the text's claims (lights most, WeekStatus least, RH_1, RH_out, RH_8, RH_3 strong) all hold |
| 12 | RF and LightGBM scatters almost identical; XGBoost similar; GBM, CatBoost differ | The paper's R² is the **squared correlation** of in-sample predictions, not the usual R²: refitting on all 3,289 hours and squaring the correlation gives LightGBM 0.831 (paper 0.831) and RF 0.963 (0.964); the usual R² of the same predictions is 0.706 and 0.930. GBM gives 0.540 vs 0.63; XGBoost and CatBoost not checked (libraries not installed locally). Usual in-sample R²: GBM 0.31, XGBoost 0.72, CatBoost 0.98: settings for this experiment are not given ("fine-tuned with optimal hyperparameter values"), so ours come from the same grid search. **On held-out test weeks every model has R² ≤ 0.05** (RF −1.99): the sensor columns describe the hours they were measured in but do not forecast |

## University Residential Complex

| Point | Article | What we do |
|---|---|---|
| Data | Not public per Table 1 | The authors publish the finished feature table only; the raw KEPCO and KMA data are not available, so its preprocessing cannot be replicated. Table 1 statistics are checked against it |
| Split | Train Sept 2015-Dec 2017, test 2018 | 20,472 / 8,760 hours, as in the authors' code |

## Tree ensembles (Tables 5-12)

| Point | Article | What we do | Evidence |
|---|---|---|---|
| Selection score | GridSearchCV, 5-fold, score not stated | Negative MAPE; KFold without shuffling | Authors' code |
| RF "auto" | Features per split: auto, sqrt, log2 | `max_features=1.0` ("auto" was removed from scikit-learn; for regression it meant all features) | Authors' code uses `None`, the same thing |
| XGBoost feature sampling | "Feature sample by tree/level/node: 0.5, 0.75, 1.0" | Only `colsample_bytree` is varied (486 settings, not 4,374) | Authors' code |
| XGBoost "gbdt" | Booster gbdt, dart | `gbtree`, `dart` | XGBoost's name for gbdt |
| XGBoost dart | – | Kept as in the grid. With XGBoost's default `rate_drop=0` no trees are dropped, so dart behaves like gbtree | Timing probe: same fit time |
| LightGBM subsample | 0.5 | Kept, but it has no effect: LightGBM only subsamples when `subsample_freq > 0`, which neither the article nor the code sets | LightGBM documentation |
| GBM quantile loss | "quantile", no quantile given | scikit-learn default `alpha=0.9` | Authors' code |
| Seeds | Random state 42 | 42 for every model | Article |
| Versions | Not stated | Kaggle's: scikit-learn 1.9.1 for the main (fixed-setting), raw-column and figure runs, 1.6.1 for the grid search; XGBoost 3.4.1, LightGBM 4.6.0, CatBoost 1.2.10, pandas 2.3.3, NumPy 2.1.3 | Recorded with each run |

The random forest on AEP reproduces Tables 8, 9, 10 and 12 to every reported digit.

### Which settings produced the published tables

The article describes a grid search with 5-fold cross-validation for each model and input set. Run as described,
it does not reproduce the published tables (AEP differences up to 16 MAPE points, 18 CVRMSE points). Training every grid setting and
comparing it with the published rows (`run_match`) showed that the published numbers come from **one fixed setting
per model and dataset, used for all three input sets**: the settings hard-coded in the authors' notebook
`explainable_AI_energy_consumption_models.ipynb` (`ensembles.AUTHORS_SETTINGS`). This is shown exactly for AEP RF and
CatBoost and dormitory LightGBM and CatBoost (all 27 values each); for the others the fixed settings come closest
but leave differences (up to 3.3 points on AEP). The main tables use these
(`run_ensembles fixed`, scikit-learn 1.9.1); the grid-search results are kept as what the described method gives.

| Tables | With the authors' settings |
|---|---|
| 5-7, 11 (dormitory) | LightGBM and CatBoost exact; GBM within 0.01; RF within 0.05; XGBoost within 0.08 |
| 8-10, 12 (AEP) | RF and CatBoost exact (all 27 values each); LightGBM internal within 0.04, others within 1.6; GBM up to 2.7; XGBoost up to 3.3 |

The authors' settings are mid-table on AEP under cross-validation *and* on the test set, so they were neither tuned
on AEP nor picked using the test set. For GBM and LightGBM they rank among the best on the dormitory data
(tuned there, then reused); for XGBoost and CatBoost they rank near the bottom even there. XGBoost 1.7.6 (before the
2.0 change of default tree method) does not bring XGBoost closer, so that hypothesis is ruled out.

### Differences from the paper and their causes (grid-search results)

| Where | Difference | Cause |
|---|---|---|
| Table 5, LightGBM | About +2 points in every cell | Cross-validation ranks `dart` (MAPE 10.11%) just ahead of `gbdt` (10.32%). Refitting the second-ranked `gbdt`, learning rate 0.01, 1,000 trees (also the setting in the authors' final-model notebook) reproduces all nine values exactly. LightGBM's dart drops trees at random, so a near-tie can flip between versions |
| Tables 5-7, XGBoost and CatBoost | XGBoost 1.24 better to 0.01 worse; CatBoost 0.81 better to 0.04 worse | Unexplained. XGBoost 1.7.6 (before the 2.0 default change) was tried and is further off; CatBoost versions not tested |
| Figure 12 | Paper's R² of 0.63-0.96 are squared correlations of in-sample predictions | See the Figure 12 row above |

## Deep networks (Tables 13-14)

Ten networks, settings from the article (Adam 0.001, batch 24, 100 epochs, seed 42) and the authors' notebook (14
units, linear activations + LeakyReLU, 24-hour input window, self-attention). We don't use the notebook's early
stopping because the article says 100 epochs. Scored two ways: the 24th forecast hour (article) and every horizon of
test-only windows (notebook). Full tables: [results/table13.md](results/table13.md), [results/table14.md](results/table14.md).

| | Dormitory (Table 13) | AEP (Table 14) |
|---|---|---|
| Best ensemble (ours = paper) | GBM MAPE 5.13 | CatBoost MAPE 41.35 |
| Our networks, MAPE (24th hour) | 6.9-8.9 | 46.9-141.6 |
| Paper's networks, MAPE | 6.4-15.1 | 34.3-50.4 |
| Networks that beat the best ensemble, paper | None, on any metric | On MAPE six (34.31-40.97 vs 41.35); none on CVRMSE or NMAE |
| Networks that beat the best ensemble, ours | None, on any metric | None, on any metric (also with early stopping) |

- **Values aren't reproduced.** Dormitory: 16 of 30 cells within 1 point (MAPE and NMAE 7 of 10, CVRMSE 2 of 10);
  our LSTM is far better than the paper's (6.98 vs 15.06), our Bi-LSTM-TCN unstable. AEP: ours worse on MAPE for 9
  of 10 networks (e.g. Bi-GRU 141.6 vs 48.2).
- **Why AEP is so much worse:** the final training loss is 0.006-0.018 (scaled; most 0.007-0.010), but test predictions range from
  -4,113 to 6,077 Wh against actual values of 210-3,200 Wh, and up to 106 of 978 are negative. With about 2,300
  training windows and 100 epochs without early stopping, the networks overfit. Their linear activations then let
  outputs run unbounded on unfamiliar inputs (the same mechanism as the dormitory Bi-LSTM-TCN at Lunar New Year). The
  notebook's early stopping on a 10% validation split probably kept the authors' networks from overfitting this way.
- **Early stopping, tested** (`run_deep --early-stopping`, the notebook's settings: last 10% of training windows
  for validation, patience 10, best weights restored;
  [results/table14_early_stopping.md](results/table14_early_stopping.md)):
  - Nine networks stop after 11-18 epochs, so their best epoch is between 1 and 8. TCN runs all 100.
  - The exploding predictions are gone: predictions now range from -258 to 3,235 Wh, with at most 6 negatives.
  - CVRMSE is now 78-92, close to the paper's 81-98 (four networks are slightly below the paper's lowest).
  - MAPE improves for 8 of 10 networks (52-73 instead of 47-142), but stays 6-35 points above the paper's 34-50 (median gap about 21). LSTM and LSTM-TCN get worse.
  - Every network is still worse than CatBoost on all three metrics.
  - So early stopping explains the explosions and the CVRMSE gap, not the whole MAPE gap. Our MAPE comes mostly from
    low-consumption hours: for GRU the mean percentage error is 84% on the lowest quarter of hours and 42% on the
    rest. A forecast that overshoots by 100 Wh at a 210 Wh hour has a 48% error for that hour. The causes of the
    remaining difference are untested (only one seed and one TensorFlow version were run).
- **Scoring question unresolved.** Every-horizon scoring is closer to Table 13 on CVRMSE (7 of 10 networks) but not on
  MAPE or NMAE (3 of 10 each); on AEP neither scoring is systematically closer either.
- Possible reasons the values differ (untested): random initialisation, GPU arithmetic, library versions (unknown for
  the authors); the notebook also doesn't run as published.

## Metrics

| Point | Article | What we do |
|---|---|---|
| Day-type groups | W, H, T in Tables 5-10 | Holi 0, Holi 1, all; the mean in CVRMSE and NMAE is taken within each group (authors' code) |
| Harmonic mean | Eq (11) | Computed from unrounded metrics; recomputing it from the rounded table values differs by at most 0.01 |

## Paper inconsistencies to mention in the report

- The Conclusions say GBM with external inputs was best on AEP; Table 12 gives CatBoost with all inputs the lowest harmonic mean (in Table 10 GBM, RF and CatBoost each lead on one metric).
- The results text says errors were lowest with external inputs. All 15 inputs give the lowest value in all 45 cells of Tables 5-7 and 38 of 45 of Tables 8-10; external inputs alone in 7.
- The paragraph on Table 14 contains garbled numbers.
- The Figure 13 text codes WeekStatus as weekend 0 / weekday 1, the reverse of Table 2.
