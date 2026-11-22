# Audit 2 (2026-10-21 to 2026-11-22): six parts, one agent at a time, no hints

Complete. Parts 1-2 ran on 2026-10-21, parts 3-6 on 2026-11-22; all fixes were applied after the last part. The agents were told not to read
`docs/audit/` and to judge the documents fresh.

## Part 1: report summary, Tables 5-12, tuning section; README results table; docs/results tables 05-12

Checked and correct: every cell of tables 05-12 (Markdown and HTML) against an independent recomputation; the
summary counts (79/135, 0.08; 55/135, 3.26); every per-model range in the README; the day groups (AEP H = weekends
only; dormitory H = 104 weekend days + 15 weekday holidays); the fixed settings against the authors' notebook; the
tuning-section claims (0.607; 11.84); Table 1; the citation.

| # | Class | Finding | Verdict | Change |
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

| # | Class | Finding | Verdict | Change |
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

## Part 3: figures (report Figures 2-13, fig12_r2, walkthroughs 05-06), run 2026-11-22 after the WeekStatus rerun

Checked and correct: every figure pair panel by panel (Figs 2-4, 8, 10, 11 exact or point-close; 5, 9 swaps as
stated; 6b/7b remark), all Figure 12 values across json, log, results file and report (RF and GBM recomputed
locally), Figure 13 colours under both codings, walkthrough 05 base values and budget code, walkthrough 06 quotes.

| # | Class | Finding | Verdict | Change |
|---|---|---|---|---|
| 1 | ERROR | Walkthrough 06 code shows the old WeekStatus line | Confirmed | Updated |
| 2 | INCONSISTENCY | `r2_score` labelled "the paper's kind of R²" | Confirmed | Relabelled; corr² line added |
| 3 | INCONSISTENCY | "Why do GBM, XGBoost, LightGBM and CatBoost differ?" | Confirmed: LightGBM matches on squared correlation | LightGBM removed |
| 4 | ERROR | XGBoost held-out "−0.66" | Confirmed: −0.6549 | −0.65 |
| 5 | OVERSTATEMENT | "The paper's R² is the squared correlation" | Confirmed: holds for RF and LightGBM (whose scatters we reproduce); XGBoost 0.805 vs 0.786 is not a match | Restricted to those two |
| 6 | UNSUPPORTED | "Can only come from in-sample scoring" | Confirmed: a random split was never tested | "comes from scoring the training hours (the 4,000 Wh training hour is in the paper's scatter)" |
| 7 | UNSUPPORTED | 2,000-hour sample "gives the same picture" | Confirmed by eye: shorter tails than the paper's 5a/5d; never tested | Stated |
| 8 | UNSUPPORTED | "far above the budget" | Confirmed: estimate not saved | "above the 30-minute budget"; 5d sampled too |
| 9 | ERROR | PDP contours "average prediction" | Confirmed: scikit-learn's recursion method omits the GBM init constant (its docstring); levels go below the dorm minimum | Relative levels |
| 10 | MISLABEL | SHAP units "Wh" for all | Confirmed: dormitory is kWh | Per dataset |
| 11 | MISQUOTE | THI and WCT "matter more than temperature" | Confirmed: not in the article; its sentence compares WCT with THI, humidity and temperature | Real quote |
| 12 | MINOR | Fig 6 "within about 21" | Confirmed: largest labelled gap 17.6 (21 is Fig 7) | "within 18" |
| 13 | MINOR | Fig 7a differences not described | Confirmed | Noted |
| 14 | MINOR | Fig 13 "top 12, bottom 3" understated | Confirmed: top 13 and bottom 5 in order | Updated |
| 15 | MINOR | Card groups 13 with "closely" | Confirmed | "13 very closely" |
| 16 | MINOR | Fig 11 axis: "test period peaks far lower" | Confirmed: test actual peak 3,200; it's the predictions (1,876 vs 3,881) | Reworded |
| 17 | MINOR | Code comment "RF 0.963" | Confirmed: 0.9638 | 0.964, two models only |

## Part 4: paper issues (report) and replication_notes.md

Checked and correct: all 11 issue quotes appear in the article; paper.py against Tables 5-14 cell by cell; every
number in the issues; the notes' data, grid, fixed-setting and deep-network numbers (recomputed); Figures 4-11 notes.

| # | Class | Finding | Verdict | Change |
|---|---|---|---|---|
| 1 | OVERSTATEMENT | Issue 5 / notes: Figure 12 R² "are in-sample squared correlations" | Confirmed: 2 of 5 models | RF and LightGBM only |
| 2 | INCONSISTENCY | Issue 11: "stopping alone isn't isolated" | Confirmed: the per-epoch run isolates it (90% data: 115.9 vs 61.2) | Updated |
| 3 | OVERSTATEMENT | Issue 1: "the tables come from one fixed setting" | Confirmed: exact for 4 of 10 model/dataset pairs | Qualified |
| 4 | OVERSTATEMENT | Issue 2: the Conclusions misread Table 12 | Confirmed: "GBM with external factors" can mean best within the external column, where it is (50.82). Stronger point: the results section says "the CatBoost model exhibited superior performance on the AEP dataset" (quote verified), so the article contradicts itself | Reframed as results vs conclusions |
| 5 | OVERSTATEMENT | Issue 3: "external inputs gave the lowest errors" is wrong | Confirmed ambiguous: all-15 includes external factors; the next paragraph says the combination was best (quote verified) | Presented as ambiguous wording |
| 6 | INCONSISTENCY | Notes: grid XGBoost/CatBoost differences "unexplained" | Confirmed: explained by the fixed settings | Cause stated |
| 7 | MINOR | XGBoost 1.7.6 "further off" | Confirmed: dormitory and AEP maximum; AEP sum about equal (28.2 vs 28.5) | Precise |
| 8 | MINOR | Seeds attributed to the article | Confirmed: article only for deep networks; ensembles from the authors' code | Fixed |
| 9 | MINOR | Quotes not word for word | Confirmed | Exact quotes |
| 10 | MINOR | Fig 13 "top 12, bottom 3" | Same as part 3 #14 | Top 13, bottom 5 |
| 11 | MINOR | Taylor "point for point" in the notes vs "very close" in the report | Confirmed | One verdict |
| 12 | MINOR | "Run as described" for XGBoost | Confirmed: only colsample_bytree varied (authors' code) | "as in the authors' grid code" |
| 13 | UNSUPPORTED | run_match tables, CV rankings, the 10.11/10.32 CV MAPEs, the dart timing probe, XAI versions: no saved evidence | Confirmed | Mark as not saved, or save the evidence |

## Part 5: "Beyond the paper" (significance, stopping) and audit1.md

Checked and correct: significance.py (Newey-West, HLN variant, t distribution, losses, bootstrap, day assignment for
both datasets); all 54 p-values reproduced by an independent implementation; every bootstrap share and interval;
stopping.py rules against Prechelt and Keras; every cell of stopping_aep.md; the tracking run against the
early-stopping run (all ten equal to 4 decimals); most numeric verdicts of audit1.md.

| # | Class | Finding | Verdict | Change |
|---|---|---|---|---|
| 1 | MISLABEL | "The article's own setup … gives a mean MAPE of 81.5" | Confirmed: 81.5 is our run of that setup; the paper's networks average 41.9; number typed, not computed | "Our run of the article's setup"; computed |
| 2 | MINOR | Published text omits the HLN factor and t distribution of the main test | Confirmed | Described |
| 3 | MINOR | Size check simulates the loss difference, not errors; 8.0% has ±1.1 Monte Carlo error | Confirmed (4,000 runs: 7.7%) | "about 8%", loss differences |
| 4 | MINOR | Bootstrap bounds printed to 2 decimals carry about ±0.5 noise on AEP | Confirmed (seed 7) | Noted |
| 5 | MINOR | "Each row is a stopping rule … from training and validation loss only" | Confirmed: fixed-epoch rows use no loss; mean columns unexplained | Reworded |
| 6 | MINOR | Stopping chart's MAPE axis capped at 200, clipping 1D-CNN and TCN | Confirmed (stopping.py) | Caption states the cap |
| 7 | INCONSISTENCY | audit1 part 2 #5: "RF 0.963" | Confirmed: stale since the WeekStatus rerun (0.964) | Note added |
| 8 | ERROR | audit1 part 3 #22: grid RF "up to 0.61" | Confirmed: 0.83 on the dormitory, where the grid chose max_features 1.0 | Corrected |
| 9 | OVERSTATEMENT | audit1 part 3 #11: "the minima are later" | Confirmed: later for 8 of 10 | Corrected |
| 10 | MINOR | audit1 part 3 #31: "tests cover split and lagged inputs only" | Confirmed: also Table 1, feature table, metrics | Corrected |
| 11 | MINOR | audit1 header date 2026-10-21 | The work ran late on 2026-10-20 into 10-21; commits are 10-20 | Date range |

## Part 6: progress log, README text, walkthroughs 01-04 and 08

Checked and correct: walkthroughs 01-03 numbers and code (end labelling, 137 mismatches, Cons_avg 3,121/3,121, Eqs
5-11, 30 harmonic means), walkthrough 04 grids and gaps, walkthrough 08 counts and figure numbers, the progress log's
numbers, README layout and commands.

| # | Class | Finding | Verdict | Change |
|---|---|---|---|---|
| 1 | ERROR | WT08 Figure 13 heading and Q&A still say colours reversed, "same numbers" | Confirmed | Updated |
| 2 | ERROR | WT08 "our bars are lower than the paper's" | Confirmed: CatBoost 0.983 > 0.932 | Per model |
| 3 | ERROR | WT08 "GBM is furthest off" | Confirmed: XGBoost (3.26, mean 1.06) | Reworded |
| 4 | ERROR | WT04 grid models "better on MAPE, worse at the peaks"; "pick underfit models" | Confirmed: better on CVRMSE too (34/45 cells); low predictions only for GBM, LightGBM | Corrected; WT03 narrowed |
| 5 | ERROR | WT08 Table 1 "count, standard deviation" | Confirmed: min, quartiles, median, mean, max | Fixed |
| 6 | MISLABEL | WT02 "AEP range" is the training range | Confirmed | Labelled |
| 7 | INCONSISTENCY | Taylor "point for point" in log overview and WT08 | Confirmed | "very close" |
| 8 | INCONSISTENCY | WT08 kind C "comes closest of all" | Confirmed: closest or second-closest | Fixed |
| 9 | INCONSISTENCY | WT08 kind B "(not tested)" | Confirmed: XGBoost 1.7.6 tried | Fixed |
| 10 | INCONSISTENCY | WT08 lists Figure 4 as kind A | Confirmed: dormitory GBM is kind B | Fixed |
| 11 | OVERSTATEMENT | README/log/WT08: Figure 12 R² "are squared correlations" | Confirmed: RF and LightGBM only | Restricted |
| 12 | UNSUPPORTED | WT08: GBM/CatBoost differ "because" settings come from our grid | Confirmed: RF and LightGBM used the same grid; untested. Also: the raw settings were chosen by cross-validation under the old coding | Reworded; provenance noted |
| 13 | UNSUPPORTED | WT04 "repository created after acceptance" | Confirmed: shallow clone; the authors' response says code was public | "current version dates from 2024-10-27" |
| 14 | OVERSTATEMENT | README: "RF and CatBoost reproduce exactly" | Confirmed: AEP only; LightGBM, CatBoost on the dormitory | Per dataset |
| 15 | OVERSTATEMENT | README: "the Conclusions contradict the tables" | Same as part 4 #4 | Results vs conclusions |
| 16 | INCONSISTENCY | Log never records audit 1 part 3 as done | Confirmed | Entry |
| 17 | MINOR | Log: networks "about 1-3 hours each" | Confirmed: 0.30-2.87 h | Fixed in a note |
| 18 | MINOR | Log: "115.9 at epoch 100" without the 90% split | Confirmed | Noted |
| 19 | MINOR | Log: Fig 11 "CatBoost" differs from the notebook | Confirmed: the notebook uses CatBoost too; the difference is training hours and order | Noted |
| 20 | MINOR | Log overview: Figure 12 "RF" only | Confirmed | RF and LightGBM |
| 21 | MINOR | WT02 Cons_avg snippet missing the empty case | Confirmed | Fixed |
| 22 | MINOR | WT02 "Cons_avg is the strongest input" | Confirmed: dormitory only | Narrowed |
| 23 | UNSUPPORTED | WT03 "Bi-GRU-TCN predicts low" | Confirmed: not shown | Reworded |
| 24 | MINOR | WT03 "article doesn't say which mean" | Confirmed: "mean of all observed values" suggests overall | Reworded |
| 25 | MINOR | WT08 "27 sensor columns" | Confirmed: includes calendar columns | Fixed |
| 26 | MINOR | "Explosions disappear" | Confirmed: still down to −258 Wh | "extreme predictions" |
| 27 | MINOR | WT08 "every number was checked" | Confirmed by the above | Removed |

## Result

Parts 1-6: 6 + 10 + 17 + 13 + 11 + 27 = 84 findings, all confirmed, none rejected; about ten were outright errors
(stale code in walkthroughs, a PDP concept, a misquote, wrong "furthest off" and "lower than the paper's"
claims), the rest overstatements, labels and wording. No reproduced number in the report's tables was wrong. The
pending additions above (Figure 13 rerun, Figure 6 issue, Figure 12 table) are done.

