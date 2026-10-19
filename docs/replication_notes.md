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
| Versions | Not stated | Kaggle's: scikit-learn 1.6.1, XGBoost 3.4.1, LightGBM 4.6.0, CatBoost 1.2.10, pandas 2.3.3, NumPy 2.1.3 | Recorded with each run |

The random forest on AEP reproduces Tables 8, 9, 10 and 12 to every reported digit.

### Which settings produced the published tables

The article describes a grid search with 5-fold cross-validation for each model and input set. Run as described,
it does not reproduce the published tables (AEP differences up to 18 points). Training every grid setting and
comparing it with the published rows (`run_match`) showed that the published numbers come from **one fixed setting
per model, used for all three input sets**: the settings hard-coded in the authors' notebook
`explainable_AI_energy_consumption_models.ipynb` (`ensembles.AUTHORS_SETTINGS`). The main tables use these
(`run_ensembles fixed`, scikit-learn 1.9.1); the grid-search results are kept as what the described method gives.

| Tables | With the authors' settings |
|---|---|
| 5-7, 11 (dormitory) | LightGBM and CatBoost exact; GBM within 0.01; RF within 0.05; XGBoost within 0.08 |
| 8-10, 12 (AEP) | RF and CatBoost exact (all 27 values each); LightGBM internal within 0.04, others within 1.6; GBM and XGBoost within 1-3 |

The authors' settings are mid-table on AEP under cross-validation *and* on the test set, so they were neither tuned
on AEP nor picked using the test set. For GBM and LightGBM they rank among the best on the dormitory data
(tuned there, then reused); for XGBoost and CatBoost they rank near the bottom even there. XGBoost 1.7.6 (before the
2.0 change of default tree method) does not bring XGBoost closer, so that hypothesis is ruled out.

### Differences from the paper and their causes (grid-search results)

| Where | Difference | Cause |
|---|---|---|
| Table 5, LightGBM | About +2 points in every cell | Cross-validation ranks `dart` (MAPE 10.11%) just ahead of `gbdt` (10.32%). Refitting the second-ranked `gbdt`, learning rate 0.01, 1,000 trees (also the setting in the authors' final-model notebook) reproduces all nine values exactly. LightGBM's dart drops trees at random, so a near-tie can flip between versions |
| Tables 5-7, XGBoost and CatBoost | Ours 0.1-0.7 points better | Likely library versions: XGBoost 2.0 changed the default `tree_method` from exact/approx to `hist`, and CatBoost defaults have changed since the paper's runs. Not yet tested |
| Figure 12 | Paper's R² of 0.63-0.96 are in-sample | See Figures 12-13 below |

## Metrics

| Point | Article | What we do |
|---|---|---|
| Day-type groups | W, H, T in Tables 5-10 | Holi 0, Holi 1, all; the mean in CVRMSE and NMAE is taken within each group (authors' code) |
| Harmonic mean | Eq (11) | Computed from unrounded metrics; recomputing it from the rounded table values differs by at most 0.01 |

## Paper inconsistencies to mention in the report

- The Conclusions say GBM with external inputs was best on AEP and that THI and WCT matter most; Tables 10 and 12 show CatBoost with all inputs is best, and the text on Figure 9 says weather matters little.
- The results text says errors were lowest with external inputs; Tables 8-10 show the opposite.
- The paragraph on Table 14 contains garbled numbers.
- The Figure 13 text codes WeekStatus as weekend 0 / weekday 1, the reverse of Table 2.
