"""Find which grid setting reproduces each row of the paper's Tables 5-10.

  python -m moon2024.run_match --dataset aep --model gbm [--out /kaggle/working]

Every setting of the model's grid is trained on the training part and scored on
the test part, for each input set. The setting whose nine values are closest to
the paper's row shows what the paper used, which can then be compared with what
cross-validation selects.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from . import data, ensembles, metrics, paper
from .features import INPUT_SETS, TARGET


def test_scores(model: str, params: dict, train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> list[float]:
    fitted = ensembles.make_estimator(model, **params).fit(train[columns], train[TARGET])
    table = metrics.scores_by_day_type(test[TARGET], fitted.predict(test[columns]), test["Holi"])
    return [table.loc[c.split("_")[1], c.split("_")[0]] for c in paper.COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=data.DATASETS, required=True)
    parser.add_argument("--model", choices=ensembles.MODELS, required=True)
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train, test = data.load_dataset(args.dataset)
    grid = ensembles.settings(args.model)
    rows = []
    for inputs, columns in INPUT_SETS.items():
        _, table = paper.ENSEMBLE_TABLES[(args.dataset, inputs)]
        target = table[args.model]
        started = time.time()
        results = Parallel(n_jobs=-1)(delayed(test_scores)(args.model, p, train, test, columns) for p in grid)
        print(f"{inputs}: {len(grid)} settings in {time.time() - started:.0f} s", flush=True)
        for setting, (params, values) in enumerate(zip(grid, results)):
            rows.append({"inputs": inputs, "setting": setting, "params": json.dumps(params),
                         "max_abs_diff": max(abs(round(v, 2) - t) for v, t in zip(values, target)),
                         **dict(zip(paper.COLUMNS, values))})

    table = pd.DataFrame(rows)
    table.to_csv(args.out / f"match_{args.dataset}_{args.model}.csv", index=False)
    closest = table.loc[table.groupby("inputs")["max_abs_diff"].idxmin()]
    print(closest[["inputs", "setting", "max_abs_diff", "params"]].to_string(index=False))


if __name__ == "__main__":
    main()
