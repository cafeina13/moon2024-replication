## Stopping rules on the AEP deep networks

Trained 100 epochs on 90% of the training windows (10% held back for validation); the test hours were scored after every epoch. Each cell: test MAPE (24th hour) at the epoch the rule picks, with that epoch in brackets. The last two columns are means over the ten networks.

| Rule | 1d-cnn | bigru | bilstm | gru | lstm | bigru-tcn | bilstm-tcn | gru-tcn | lstm-tcn | tcn | Mean MAPE | Mean CVRMSE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Epoch 100 (the article's epoch count, on 90% of the data) | 348.8 (100) | 165.4 (100) | 124.0 (100) | 81.7 (100) | 81.4 (100) | 68.4 (100) | 58.8 (100) | 79.4 (100) | 84.3 (100) | 66.9 (100) | 115.9 | 238.3 |
| Patience 10 (authors' notebook) | 65.6 (5) | 59.2 (5) | 72.8 (1) | 52.1 (7) | 60.0 (3) | 59.8 (8) | 63.8 (7) | 54.4 (8) | 65.6 (1) | 58.7 (99) | 61.2 | 83.1 |
| Lowest validation loss, all 100 epochs | 99.7 (40) | 93.5 (32) | 72.8 (1) | 77.4 (52) | 50.1 (20) | 59.8 (8) | 55.4 (94) | 62.3 (55) | 53.3 (55) | 58.7 (99) | 68.3 | 96.3 |
| Epoch 10 | 74.0 (10) | 58.5 (10) | 72.4 (10) | 57.4 (10) | 72.1 (10) | 59.4 (10) | 61.9 (10) | 55.8 (10) | 62.0 (10) | 566.0 (10) | 114.0 | 130.3 |
| Epoch 25 | 111.0 (25) | 81.5 (25) | 65.5 (25) | 68.5 (25) | 60.9 (25) | 62.8 (25) | 57.9 (25) | 57.0 (25) | 59.8 (25) | 216.6 (25) | 84.1 | 108.3 |
| Epoch 50 | 279.1 (50) | 114.7 (50) | 83.5 (50) | 73.1 (50) | 71.6 (50) | 66.4 (50) | 67.5 (50) | 63.7 (50) | 59.1 (50) | 129.9 (50) | 100.9 | 181.6 |
| GL > 5: validation loss 5% above its best | 65.6 (5) | 59.2 (5) | 72.8 (1) | 60.8 (4) | 60.0 (3) | 59.8 (8) | 53.9 (26) | 54.4 (8) | 65.6 (1) | 349.6 (14) | 90.2 | 107.4 |
| PQ > 1: GL / training progress | 65.6 (5) | 93.5 (32) | 72.8 (1) | 52.1 (7) | 50.1 (20) | 59.8 (8) | 66.7 (54) | 62.3 (34) | 45.9 (29) | 144.9 (35) | 71.4 | 94.6 |
| UP2: validation loss rose over two strips | 99.7 (40) | 59.2 (5) | 72.8 (1) | 52.1 (7) | 60.0 (3) | 59.8 (8) | 53.9 (26) | 62.3 (55) | 46.6 (15) | 58.7 (99) | 62.5 | 89.7 |
| Best test epoch (not a rule; uses the test set) | 55.6 (3) | 50.4 (6) | 57.0 (17) | 49.9 (1) | 43.7 (18) | 50.0 (21) | 51.0 (21) | 50.9 (15) | 45.1 (36) | 57.0 (90) | 51.1 | 83.8 |
