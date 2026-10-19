# Replicating Moon et al. (2024)

A from-scratch replication of

> Moon, J., Maqsood, M., So, D., Baik, S. W., Rho, S., & Nam, Y. (2024). Advancing ensemble learning techniques
> for residential building electricity consumption forecasting: Insight from explainable artificial intelligence.
> *PLOS ONE*, 19(11), e0307654. https://doi.org/10.1371/journal.pone.0307654

on both of its datasets: the UCI Appliances Energy Prediction data and the University Residential Complex data
published in the [authors' repository](https://github.com/sodayeong/PLOS-ONE_Github).
Every choice the article leaves open is recorded in [docs/replication_notes.md](docs/replication_notes.md).

## Layout

| Path | Contents |
|---|---|
| `src/moon2024/` | Data loading, the 15 input variables, metrics |
| `tests/` | Table 1, the authors' feature table, the split, leakage and metric checks |
| `kaggle/` | One folder per Kaggle notebook; heavy steps run there, not locally |
| `docs/` | Replication notes |

## Running

Data lives in a Kaggle notebook output (`kaggle/get-data`). Locally, put the three CSVs under `data/` (or point
`MOON_DATA_DIR` at them), then:

```
pip install -e ".[test]"
pytest
```
