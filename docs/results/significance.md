## Significance of the differences (beyond the paper)

The paper ranks models by their scores without tests. Here: Diebold-Mariano tests (Newey-West variance, 24 lags, two-sided, not corrected for multiple comparisons) and a day-block bootstrap with 2000 draws. APE, SE, AE: absolute percentage, squared and absolute error, the losses behind MAPE, CVRMSE and NMAE. "Ranks first": share of bootstrap draws in which the model has the lowest harmonic mean. p-values in brackets: the textbook Harvey-Leybourne-Newbold variance (equal weights, 23 lags). In a simulation with strongly correlated errors (AR(1), coefficient 0.8, 978 hours, no real difference; `size_check`) the main test gave p < 0.05 in 8.0% of 400 runs instead of 5%, so p-values near 0.05 are weak evidence. The reference model is the best on this same test set, which also favours finding a difference.

## University Residential Complex

### Ensembles, all 15 inputs (Table 7)

8760 test hours, 365 days. Best by harmonic mean: **GBM**. p-values: Diebold-Mariano against the best model, per loss.

| Model | HM | p (APE) | p (SE) | p (AE) | Ranks first | HM minus best, 95% interval |
|---|---|---|---|---|---|---|
| GBM | 5.86 | – | – | – | 73% | – |
| CatBoost | 5.91 | 0.985 (0.988) | 0.232 (0.333) | 0.435 (0.513) | 25% | -0.10 to +0.22 |
| LightGBM | 5.97 | 0.018 (0.042) | 0.106 (0.159) | 0.099 (0.142) | 2% | +0.00 to +0.22 |
| RF | 6.23 | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0% | +0.24 to +0.50 |
| XGBoost | 6.24 | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0% | +0.26 to +0.51 |

### Input sets for GBM

8760 test hours, 365 days. Best by harmonic mean: **GBM, total**. p-values: Diebold-Mariano against the best model, per loss.

| Model | HM | p (APE) | p (SE) | p (AE) | Ranks first | HM minus best, 95% interval |
|---|---|---|---|---|---|---|
| GBM, total | 5.86 | – | – | – | 100% | – |
| GBM, internal | 6.71 | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0% | +0.59 to +1.11 |
| GBM, external | 12.88 | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0% | +6.52 to +7.56 |

### Best ensemble against the best deep network (gru, 24th hour)

8760 test hours, 365 days. Best by harmonic mean: **GBM**. p-values: Diebold-Mariano against the best model, per loss.

| Model | HM | p (APE) | p (SE) | p (AE) | Ranks first | HM minus best, 95% interval |
|---|---|---|---|---|---|---|
| GBM | 5.86 | – | – | – | 100% | – |
| gru | 7.68 | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0% | +1.32 to +2.32 |

## Appliances Energy Prediction

### Ensembles, all 15 inputs (Table 10)

978 test hours, 41 days. Best by harmonic mean: **GBM**. p-values: Diebold-Mariano against the best model, per loss.

| Model | HM | p (APE) | p (SE) | p (AE) | Ranks first | HM minus best, 95% interval |
|---|---|---|---|---|---|---|
| GBM | 47.45 | – | – | – | 67% | – |
| CatBoost | 48.23 | 0.322 (0.342) | 0.324 (0.355) | 0.448 (0.468) | 32% | -1.85 to +3.57 |
| RF | 50.40 | 0.000 (0.000) | 0.008 (0.004) | 0.051 (0.036) | 0% | +0.67 to +5.53 |
| LightGBM | 50.68 | 0.002 (0.002) | 0.948 (0.952) | 0.018 (0.026) | 0% | +0.41 to +6.06 |
| XGBoost | 56.34 | 0.000 (0.000) | 0.142 (0.153) | 0.000 (0.001) | 0% | +4.06 to +14.13 |

### Input sets for GBM

978 test hours, 41 days. Best by harmonic mean: **GBM, total**. p-values: Diebold-Mariano against the best model, per loss.

| Model | HM | p (APE) | p (SE) | p (AE) | Ranks first | HM minus best, 95% interval |
|---|---|---|---|---|---|---|
| GBM, total | 47.45 | – | – | – | 87% | – |
| GBM, internal | 49.22 | 0.565 (0.617) | 0.041 (0.024) | 0.296 (0.322) | 13% | -1.40 to +4.71 |
| GBM, external | 51.59 | 0.012 (0.005) | 0.041 (0.029) | 0.002 (0.001) | 0% | +0.99 to +7.07 |

### Best ensemble against the best deep network (lstm-tcn, 24th hour)

978 test hours, 41 days. Best by harmonic mean: **GBM**. p-values: Diebold-Mariano against the best model, per loss.

| Model | HM | p (APE) | p (SE) | p (AE) | Ranks first | HM minus best, 95% interval |
|---|---|---|---|---|---|---|
| GBM | 47.45 | – | – | – | 99% | – |
| lstm-tcn | 55.44 | 0.095 (0.175) | 0.044 (0.077) | 0.016 (0.044) | 1% | +1.36 to +15.98 |
