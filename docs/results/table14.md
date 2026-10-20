## Table 14: deep networks, Appliances Energy Prediction

Each metric: ours scored on the 24th forecast hour (the article) / ours scored on every hour of every test window (the authors' notebook) / paper, in %. The last row is the best ensemble (all 15 inputs).

| Model | MAPE | CVRMSE | NMAE |
|---|---|---|---|
| lstm | – / – / 38.50 | – / – / 88.84 | – / – / 50.11 |
| bilstm | – / – / 37.61 | – / – / 87.31 | – / – / 48.66 |
| gru | – / – / 45.88 | – / – / 98.15 | – / – / 58.70 |
| bigru | – / – / 48.17 | – / – / 92.46 | – / – / 55.78 |
| 1d-cnn | – / – / 39.95 | – / – / 87.84 | – / – / 49.66 |
| tcn | – / – / 39.02 | – / – / 90.09 | – / – / 51.00 |
| lstm-tcn | – / – / 50.39 | – / – / 96.75 | – / – / 55.36 |
| bilstm-tcn | – / – / 40.97 | – / – / 81.42 | – / – / 46.81 |
| gru-tcn | – / – / 44.13 | – / – / 89.13 | – / – / 52.77 |
| bigru-tcn | – / – / 34.31 | – / – / 86.19 | – / – / 46.58 |
| catboost | 41.35 / 41.35 (+0.00) | 68.69 / 68.69 (+0.00) | 42.61 / 42.61 (+0.00) |
