# Replicating Moon et al. (2024)

A from-scratch replication of

> Moon, J., Maqsood, M., So, D., Baik, S. W., Rho, S., & Nam, Y. (2024). Advancing ensemble learning techniques
> for residential building electricity consumption forecasting: Insight from explainable artificial intelligence.
> *PLOS ONE*, 19(11), e0307654. https://doi.org/10.1371/journal.pone.0307654

on both of its datasets: the UCI Appliances Energy Prediction (AEP) data and the University Residential Complex
(dormitory) data published in the [authors' repository](https://github.com/sodayeong/PLOS-ONE_Github).
Every choice the article leaves open is recorded in [docs/replication_notes.md](docs/replication_notes.md); the
session-by-session history is in [docs/progress_log.md](docs/progress_log.md).

## Results

| Part of the paper | Match | Where |
|---|---|---|
| Data, hourly table, 15 inputs (Tables 1-2) | Exact: Table 1 for both datasets; our inputs equal the authors' feature table on all 3,289 hours | `tests/` |
| Tree ensembles, dormitory (Tables 5-7, 11) | LightGBM, CatBoost exact; GBM within 0.01; RF within 0.05; XGBoost within 0.08 | [docs/results](docs/results) |
| Tree ensembles, AEP (Tables 8-10, 12) | RF, CatBoost exact; LightGBM within 0.04; GBM, XGBoost within 1-3 points | [docs/results](docs/results) |
| Taylor diagrams (Figs 2-3) | Point for point | [docs/figures](docs/figures) |
| SHAP, feature importance, PDP (Figs 4-11) | Figs 4, 8, 10, 11 exact; 5, 9 near-exact; 6-7 close | [docs/figures](docs/figures) |
| Raw-column experiment (Figs 12-13) | Fig 13 and RF in Fig 12 very close | [docs/figures](docs/figures) |
| Deep networks (Tables 13-14) | Conclusion holds (the best ensemble beats every network); values close on the dormitory data, much worse on AEP | [docs/results](docs/results) |

**Findings beyond the numbers**

- **The published tables don't come from the tuning the article describes.** A 5-fold grid search per model and
  input set gives AEP errors up to 18 MAPE points away from the paper. The tables come from one fixed setting per
  model, hard-coded in the authors' notebook and reused for every input set. With those settings the tables
  reproduce (CatBoost on AEP matches all 27 values exactly).
- **Figure 12's R² values are in-sample.** The models reach R² up to 0.96 on the hours they were trained on. On
  held-out weeks every model scores R² ≤ 0.05.
- **Figure 11 is CatBoost on the training hours**, not GBM on the test set as the text says.
- **The Conclusions contradict the tables.** For example, they name GBM with external inputs as best on AEP, while
  Tables 10 and 12 show CatBoost with all 15 inputs.
- **Deep networks on AEP overfit** when trained for the article's 100 epochs. The authors' notebook uses early
  stopping, which the article doesn't mention.

## Layout

| Path | Contents |
|---|---|
| `src/moon2024/data.py`, `features.py` | Hourly tables, the 15 input variables, the train/test split |
| `src/moon2024/metrics.py` | MAPE, CVRMSE, NMAE, harmonic mean, by day type |
| `src/moon2024/ensembles.py`, `run_ensembles.py`, `run_match.py` | Grid search, the authors' fixed settings, the search for which setting the paper used |
| `src/moon2024/run_xai.py`, `taylor.py`, `run_raw_figures.py` | Figures 2-13 |
| `src/moon2024/run_deep.py` | The ten deep networks (Tables 13-14) |
| `src/moon2024/paper.py`, `compare.py` | The published tables, and ours next to them |
| `tests/` | Table 1, the authors' feature table, the split, leakage and metric checks |
| `kaggle/` | Data download and one generated folder per Kaggle job; heavy steps run there, not locally |
| `docs/` | Replication notes, progress log, results tables, figures |

## Running

Data lives in a Kaggle notebook output (`kaggle/get-data`). Locally, put the three CSVs under `data/` (or point
`MOON_DATA_DIR` at them), then:

```
pip install -e ".[test]"
pytest
```

Heavy steps run on Kaggle: `python kaggle/make_job.py SLUG [--gpu] -- "run_ensembles fixed --dataset aep --model catboost"`
writes a job pinned to the current pushed commit, and `kaggle kernels push -p kaggle/jobs/SLUG` runs it. The
published RF results need scikit-learn 1.9.1.
