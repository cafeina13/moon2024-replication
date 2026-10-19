"""Command line for the ensemble experiments (Tables 5-12), run on Kaggle.

  probe     time one fit of each model's slowest settings, to plan the shards
  search    cross-validate one shard of a model's grid on all three input sets
  finalize  pick the best setting per input set, refit on the training part, predict the test part
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from . import data, ensembles
from .features import INPUT_SETS, TARGET

# The costliest settings of each model, for the timing probe
SLOWEST = {
    "rf": {"n_estimators": 128, "max_features": 1.0},
    "gbm": {"n_estimators": 500, "learning_rate": 0.1, "max_depth": 10, "loss": "quantile"},
    "xgboost": {"n_estimators": 1000, "learning_rate": 0.1, "max_depth": 10, "subsample": 1.0,
                "colsample_bytree": 1.0, "booster": "dart"},
    "xgboost-gbtree": {"n_estimators": 1000, "learning_rate": 0.1, "max_depth": 10, "subsample": 1.0,
                       "colsample_bytree": 1.0, "booster": "gbtree"},
    "lightgbm": {"n_estimators": 1500, "learning_rate": 0.1, "num_leaves": 64, "subsample": 0.5,
                 "colsample_bytree": 1.0, "boosting_type": "dart"},
    "catboost": {"learning_rate": 0.1, "depth": 10, "l2_leaf_reg": 1},
}


def versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    found = {}
    for package in ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost"]:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = None
    return found


def probe(out: Path) -> None:
    timings = []
    for dataset in data.DATASETS:
        train, _ = data.load_dataset(dataset)
        X, y = train[INPUT_SETS["total"]], train[TARGET]
        for name, params in SLOWEST.items():
            model = name.split("-")[0]
            started = time.time()
            ensembles.make_estimator(model, **params).fit(X, y)
            seconds = time.time() - started
            timings.append({"dataset": dataset, "probe": name, "seconds": round(seconds, 2)})
            print(f"{dataset} {name}: {seconds:.1f} s", flush=True)
    (out / "probe.json").write_text(json.dumps({"versions": versions(), "timings": timings}, indent=2))


def cv_path(out: Path, dataset: str, model: str, inputs: str, index: int, count: int) -> Path:
    return out / f"cv_{dataset}_{model}_{inputs}_shard{index}of{count}.csv"


def search(dataset: str, model: str, index: int, count: int, out: Path, inputs: list[str] | None) -> None:
    train, _ = data.load_dataset(dataset)
    sets = data.input_sets(dataset)
    chosen = ensembles.shard(model, index, count)
    for name in inputs or sets:
        path = cv_path(out, dataset, model, name, index, count)
        if path.exists():
            print(f"skip {path.name}: already done", flush=True)
            continue
        table = ensembles.search(model, chosen, train[sets[name]], train[TARGET])
        table["params"] = table["params"].map(json.dumps)
        table.to_csv(path, index=False)
    (out / f"versions_{dataset}_{model}_shard{index}of{count}.json").write_text(json.dumps(versions(), indent=2))


def finalize(dataset: str, model: str, cv_dir: Path, out: Path) -> None:
    train, test = data.load_dataset(dataset)
    chosen = {}
    for name, columns in data.input_sets(dataset).items():
        files = sorted(cv_dir.rglob(f"cv_{dataset}_{model}_{name}_shard*.csv"))
        table = pd.concat(pd.read_csv(f) for f in files)
        assert table["setting"].is_unique and len(table) == len(ensembles.settings(model)), (
            f"{dataset} {model} {name}: {len(table)} of {len(ensembles.settings(model))} settings found"
        )
        table["params"] = table["params"].map(json.loads)
        params = ensembles.best(table)
        chosen[name] = params
        fitted = ensembles.make_estimator(model, **params).fit(train[columns], train[TARGET])
        pd.DataFrame({
            "Date": test.index, "Holi": test["Holi"].to_numpy() if "Holi" in test else test["WeekStatus"].rsub(1).to_numpy(),
            "y": test[TARGET].to_numpy(), "y_hat": fitted.predict(test[columns]),
        }).to_csv(out / f"pred_{dataset}_{model}_{name}.csv", index=False)
        print(f"{dataset} {model} {name}: best {params}", flush=True)
    (out / f"best_{dataset}_{model}.json").write_text(json.dumps({"params": chosen, "versions": versions()}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["probe", "search", "finalize"])
    parser.add_argument("--dataset", choices=[*data.DATASETS, data.RAW_DATASET])
    parser.add_argument("--model", choices=ensembles.MODELS)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--inputs", nargs="+", help="input sets to search (default: all of the dataset's)")
    parser.add_argument("--cv-dir", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.command == "probe":
        probe(args.out)
    elif args.command == "search":
        search(args.dataset, args.model, args.shard, args.of, args.out, args.inputs)
    else:
        finalize(args.dataset, args.model, args.cv_dir, args.out)


if __name__ == "__main__":
    main()
