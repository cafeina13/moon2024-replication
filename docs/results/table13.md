## Table 13: deep networks, University Residential Complex

Each metric: ours scored on the 24th forecast hour (the article) / ours scored on every hour of every test window (the authors' notebook) / paper, in %. The last row is the best ensemble (all 15 inputs).

| Model | MAPE | CVRMSE | NMAE |
|---|---|---|---|
| lstm | 6.98 / 6.33 / 15.06 | 10.10 / 9.10 / 18.28 | 7.14 / 6.53 / 14.48 |
| bilstm | 7.20 / 6.64 / 7.87 | 10.75 / 9.98 / 10.06 | 7.39 / 6.89 / 8.00 |
| gru | 6.96 / 6.26 / 8.54 | 9.73 / 8.97 / 11.52 | 6.92 / 6.39 / 8.62 |
| bigru | 7.05 / 6.41 / 6.41 | 10.17 / 9.30 / 8.51 | 7.21 / 6.63 / 6.54 |
| 1d-cnn | 6.99 / 6.08 / 6.56 | 9.91 / 8.59 / 8.74 | 7.02 / 6.14 / 6.73 |
| tcn | 7.26 / 6.43 / 6.36 | 9.80 / 8.82 / 8.19 | 7.23 / 6.48 / 6.37 |
| lstm-tcn | 6.93 / 6.17 / 6.63 | 9.92 / 8.79 / 8.73 | 7.02 / 6.28 / 6.78 |
| bilstm-tcn | 8.88 / 7.74 / 7.14 | 31.55 / 31.57 / 9.23 | 8.35 / 7.31 / 7.18 |
| gru-tcn | 6.94 / 6.17 / 7.01 | 10.20 / 9.02 / 9.13 | 7.09 / 6.35 / 7.17 |
| bigru-tcn | 6.86 / 6.15 / 6.92 | 10.09 / 9.03 / 9.16 | 7.03 / 6.32 / 7.14 |
| gbm | 5.13 / 5.13 (+0.00) | 7.68 / 7.67 (+0.01) | 5.35 / 5.35 (+0.00) |
