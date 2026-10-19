"""The five tree ensembles of the article's Table 4 and their grid search.

Grids follow Table 4. Where it is ambiguous the authors' code decides:
- RF "auto" features per split is max_features=1.0 (the regression meaning of "auto").
- XGBoost "feature sample by tree/level/node" is varied through colsample_bytree only,
  and "gbdt" is XGBoost's "gbtree".
- Selection uses 5-fold KFold without shuffling and negative MAPE, as in GridSearchCV
  with scoring="neg_mean_absolute_percentage_error".
"""

import time

import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterGrid

SEED = 42
CV_FOLDS = 5
SCORING = "neg_mean_absolute_percentage_error"

GRIDS = {
    "rf": {"n_estimators": [128], "max_features": [1.0, "sqrt", "log2"]},
    "gbm": {
        "n_estimators": [100, 250, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [5, 10],
        "loss": ["quantile", "huber"],
    },
    "xgboost": {
        "n_estimators": [250, 500, 1000],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [6, 8, 10],
        "subsample": [0.5, 0.75, 1.0],
        "colsample_bytree": [0.5, 0.75, 1.0],
        "booster": ["gbtree", "dart"],
    },
    "lightgbm": {
        "n_estimators": [1000, 1500],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [64],
        "subsample": [0.5],
        "colsample_bytree": [1.0],
        "boosting_type": ["gbdt", "dart"],
    },
    "catboost": {"learning_rate": [0.03, 0.1], "depth": [4, 6, 10], "l2_leaf_reg": [1, 3, 5, 7, 9]},
}
MODELS = list(GRIDS)


def make_estimator(model: str, **params):
    """A fresh estimator with the authors' fixed settings plus the given grid values.

    Each estimator runs single-threaded; parallelism comes from fitting grid
    settings and folds side by side.
    """
    if model == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=SEED, n_jobs=1, **params)
    if model == "gbm":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(random_state=SEED, **params)
    if model == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=SEED, objective="reg:squarederror", n_jobs=1, **params)
    if model == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=SEED, n_jobs=1, verbose=-1, **params)
    if model == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(random_seed=SEED, thread_count=1, verbose=0, allow_writing_files=False, **params)
    raise ValueError(f"unknown model {model!r}")


def settings(model: str) -> list[dict]:
    """Every grid setting of a model, in GridSearchCV's order."""
    return list(ParameterGrid(GRIDS[model]))


def shard(model: str, index: int, count: int) -> list[dict]:
    """Every count-th setting starting at index, so slow settings spread across shards."""
    return settings(model)[index::count]


def search(model: str, chosen: list[dict], X: pd.DataFrame, y: pd.Series, n_jobs: int = -1) -> pd.DataFrame:
    """Cross-validated score of each chosen setting, one row per setting."""
    grid = [{key: [value] for key, value in params.items()} for params in chosen]
    gs = GridSearchCV(make_estimator(model), grid, cv=CV_FOLDS, scoring=SCORING, n_jobs=n_jobs, refit=False)
    started = time.time()
    gs.fit(X, y)
    print(f"{model}: {len(chosen)} settings x {CV_FOLDS} folds in {time.time() - started:.0f} s", flush=True)
    keep = ["params", "mean_test_score", "std_test_score", "mean_fit_time"]
    table = pd.DataFrame(gs.cv_results_)[keep]
    table["params"] = table["params"].map(lambda p: {k: p[k] for k in sorted(p)})
    table["setting"] = [settings(model).index(p) for p in chosen]
    return table


def best(table: pd.DataFrame) -> dict:
    """The setting with the highest mean score; ties go to the earliest in grid order, like GridSearchCV."""
    ranked = table.sort_values(["mean_test_score", "setting"], ascending=[False, True])
    return ranked.iloc[0]["params"]
