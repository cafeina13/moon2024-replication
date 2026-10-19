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
