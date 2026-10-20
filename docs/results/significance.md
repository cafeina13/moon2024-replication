## Significance of the differences (beyond the paper)

The paper ranks models by their scores without tests. Here: Diebold-Mariano tests (Newey-West variance, 24 lags, two-sided, not corrected for multiple comparisons) and a day-block bootstrap with 2000 draws. APE, SE, AE: absolute percentage, squared and absolute error, the losses behind MAPE, CVRMSE and NMAE. "Ranks first": share of bootstrap draws in which the model has the lowest harmonic mean. In a simulation with strongly correlated errors (AR(1), coefficient 0.8, 978 hours, no real difference) the test gave p < 0.05 in 8% of 400 runs instead of 5%, so p-values near 0.05 are weak evidence.

## University Residential Complex

### Ensembles, all 15 inputs (Table 7)

8760 test hours, 366 days. Best by harmonic mean: **GBM**. p-values: Diebold-Mariano against the best model, per loss.

| Model | HM | p (APE) | p (SE) | p (AE) | Ranks first | HM minus best, 95% interval |
|---|---|---|---|---|---|---|
| GBM | 5.86 | – | – | – | 73% | – |
| CatBoost | 5.91 | 0.985 | 0.232 | 0.435 | 25% | -0.10 to +0.21 |
| LightGBM | 5.97 | 0.018 | 0.106 | 0.099 | 2% | +0.00 to +0.22 |
| RF | 6.23 | 0.000 | 0.000 | 0.000 | 0% | +0.25 to +0.50 |
| XGBoost | 6.24 | 0.000 | 0.000 | 0.000 | 0% | +0.27 to +0.51 |

### Input sets for GBM

8760 test hours, 366 days. Best by harmonic mean: **GBM, total**. p-values: Diebold-Mariano against the best model, per loss.

| Model | HM | p (APE) | p (SE) | p (AE) | Ranks first | HM minus best, 95% interval |
|---|---|---|---|---|---|---|
| GBM, total | 5.86 | – | – | – | 100% | – |
| GBM, internal | 6.71 | 0.000 | 0.000 | 0.000 | 0% | +0.60 to +1.12 |
| GBM, external | 12.88 | 0.000 | 0.000 | 0.000 | 0% | +6.48 to +7.55 |

### Best ensemble against the best deep network (gru, 24th hour)

8760 test hours, 366 days. Best by harmonic mean: **GBM**. p-values: Diebold-Mariano against the best model, per loss.

| Model | HM | p (APE) | p (SE) | p (AE) | Ranks first | HM minus best, 95% interval |
|---|---|---|---|---|---|---|
| GBM | 5.86 | – | – | – | 100% | – |
| gru | 7.68 | 0.000 | 0.000 | 0.000 | 0% | +1.35 to +2.35 |

## Appliances Energy Prediction

### Ensembles, all 15 inputs (Table 10)

978 test hours, 41 days. Best by harmonic mean: **GBM**. p-values: Diebold-Mariano against the best model, per loss.

| Model | HM | p (APE) | p (SE) | p (AE) | Ranks first | HM minus best, 95% interval |
|---|---|---|---|---|---|---|
| GBM | 47.45 | – | – | – | 67% | – |
| CatBoost | 48.23 | 0.322 | 0.324 | 0.448 | 32% | -1.85 to +3.57 |
| RF | 50.40 | 0.000 | 0.008 | 0.051 | 0% | +0.67 to +5.53 |
| LightGBM | 50.68 | 0.002 | 0.948 | 0.018 | 0% | +0.41 to +6.06 |
| XGBoost | 56.34 | 0.000 | 0.142 | 0.000 | 0% | +4.06 to +14.13 |

### Input sets for GBM

978 test hours, 41 days. Best by harmonic mean: **GBM, total**. p-values: Diebold-Mariano against the best model, per loss.

| Model | HM | p (APE) | p (SE) | p (AE) | Ranks first | HM minus best, 95% interval |
|---|---|---|---|---|---|---|
| GBM, total | 47.45 | – | – | – | 87% | – |
| GBM, internal | 49.22 | 0.565 | 0.041 | 0.296 | 13% | -1.40 to +4.71 |
| GBM, external | 51.59 | 0.012 | 0.041 | 0.002 | 0% | +0.99 to +7.07 |

### Best ensemble against the best deep network (lstm-tcn, 24th hour)

978 test hours, 41 days. Best by harmonic mean: **GBM**. p-values: Diebold-Mariano against the best model, per loss.

| Model | HM | p (APE) | p (SE) | p (AE) | Ranks first | HM minus best, 95% interval |
|---|---|---|---|---|---|---|
| GBM | 47.45 | – | – | – | 99% | – |
| lstm-tcn | 55.44 | 0.095 | 0.044 | 0.016 | 1% | +1.36 to +15.98 |
