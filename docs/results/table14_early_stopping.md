## Table 14 with early stopping: deep networks, Appliances Energy Prediction

Each metric: ours scored on the 24th forecast hour (the article) / ours scored on every hour of every test window (the authors' notebook) / paper, in %. Trained the authors' notebook way: the last 10% of training windows held out for validation, patience 10, best weights restored (9 of 10 networks stopped after 11-18 epochs; TCN ran all 100 and was restored to its best epoch, 99: its score equals epoch 99 of the per-epoch run). The last row is the best ensemble (all 15 inputs).

| Model | MAPE | CVRMSE | NMAE |
|---|---|---|---|
| lstm | 60.01 / 59.53 / 38.50 | 80.17 / 80.44 / 88.84 | 55.41 / 55.14 / 50.11 |
| bilstm | 72.79 / 71.74 / 37.61 | 92.32 / 91.78 / 87.31 | 63.19 / 62.44 / 48.66 |
| gru | 52.12 / 52.00 / 45.88 | 79.34 / 79.22 / 98.15 | 51.25 / 51.09 / 58.70 |
| bigru | 59.23 / 59.91 / 48.17 | 78.16 / 79.26 / 92.46 | 53.97 / 54.69 / 55.78 |
| 1d-cnn | 65.60 / 62.58 / 39.95 | 81.69 / 80.45 / 87.84 | 58.25 / 56.32 / 49.66 |
| tcn | 58.69 / 75.99 / 39.02 | 82.87 / 86.38 / 90.09 | 55.53 / 64.28 / 51.00 |
| lstm-tcn | 65.62 / 67.84 / 50.39 | 86.83 / 86.22 / 96.75 | 58.53 / 58.58 / 55.36 |
| bilstm-tcn | 63.77 / 60.91 / 40.97 | 88.98 / 86.76 / 81.42 | 60.46 / 57.79 / 46.81 |
| gru-tcn | 54.42 / 60.27 / 44.13 | 82.16 / 78.20 / 89.13 | 53.51 / 54.24 / 52.77 |
| bigru-tcn | 59.78 / 54.73 / 34.31 | 78.88 / 78.45 / 86.19 | 54.67 / 52.41 / 46.58 |
| catboost | 41.35 / 41.35 (+0.00) | 68.69 / 68.69 (+0.00) | 42.61 / 42.61 (+0.00) |
