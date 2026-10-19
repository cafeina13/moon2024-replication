"""Figures 4-11: feature importance, SHAP and partial dependence, run on Kaggle.

  python -m moon2024.run_xai [--best-dir /kaggle/input] [--out /kaggle/working]

Each model is the best setting for all 15 inputs (from run_ensembles finalize),
refitted on the training part. As in the authors' code, SHAP values are computed
on the training part, except Figure 11, which the article draws on the test set.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import data, ensembles, paper
from .compare import DATASET_NAMES as DATASET_TITLES
from .features import TARGET, TOTAL

SEED = 42
# Above this many estimated minutes, SHAP values are computed on a random sample of training hours
SHAP_BUDGET_MINUTES = 30
SAMPLE_ROWS = 2000
DECISION_PLOT_ROWS = 100  # authors' code: the first 100 test hours

SUMMARY_FIGURES = {
    ("urc", "gbm"): "fig04b", ("urc", "rf"): "fig05a", ("urc", "xgboost"): "fig05b",
    ("urc", "lightgbm"): "fig05c", ("urc", "catboost"): "fig05d",
    ("aep", "catboost"): "fig08b", ("aep", "rf"): "fig09a", ("aep", "gbm"): "fig09b",
    ("aep", "xgboost"): "fig09c", ("aep", "lightgbm"): "fig09d",
}
IMPORTANCE_FIGURES = {("urc", "gbm"): "fig04a", ("aep", "catboost"): "fig08a"}
DEPENDENCE_FIGURES = [
    # (figure, dataset, model, feature, colour by)
    ("fig06b", "urc", "gbm", "Holi", "Cons_avg"),
    ("fig07b", "urc", "gbm", "THI", "Cons_avg"),
    ("fig10a", "aep", "catboost", "Hour_x", "Hour_y"),
    ("fig10b", "aep", "catboost", "Holi", "Hour_y"),
]
PDP_FIGURES = [("fig06a", "urc", "gbm", ("Holi", "Cons_avg")), ("fig07a", "urc", "gbm", ("THI", "Cons_avg"))]


def best_params(best_dir: Path, dataset: str, model: str) -> dict:
    (path,) = best_dir.rglob(f"best_{dataset}_{model}.json")
    return json.loads(path.read_text())["params"]["total"]


def save(fig_name: str, title: str, out: Path) -> None:
    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out / f"{fig_name}.png", dpi=200, bbox_inches="tight")
    plt.close("all")
    print(f"wrote {fig_name}", flush=True)


def explain(estimator, X: pd.DataFrame, label: str, log: dict) -> tuple[shap.Explanation, pd.DataFrame]:
    """SHAP values on X, or on a random sample of it when the full set would exceed the budget."""
    explainer = shap.TreeExplainer(estimator)
    started = time.time()
    explainer(X.iloc[:50])
    minutes = (time.time() - started) / 50 * len(X) / 60
    if minutes > SHAP_BUDGET_MINUTES:
        X = X.sample(SAMPLE_ROWS, random_state=SEED).sort_index()
    started = time.time()
    values = explainer(X)
    log[label] = {"rows": len(X), "estimated_full_minutes": round(minutes, 1), "seconds": round(time.time() - started, 1)}
    print(f"{label}: SHAP on {len(X)} rows in {time.time() - started:.0f} s", flush=True)
    return values, X


def importance_plot(estimator, model: str, fig_name: str, title: str, out: Path) -> None:
    scores = estimator.get_feature_importance() if model == "catboost" else estimator.feature_importances_
    order = pd.Series(scores, index=TOTAL).sort_values()
    plt.figure(figsize=(6, 5))
    plt.barh(order.index, order.values, color="#3b6ea8")
    plt.xlabel("Feature importance" + (" (CatBoost, %)" if model == "catboost" else " (impurity decrease)"))
    save(fig_name, title, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--best-dir", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    log = {"shap": shap.__version__}
    for dataset in data.DATASETS:
        train, test = data.load_dataset(dataset)
        X, y = train[TOTAL], train[TARGET]
        for model in ensembles.MODELS:
            label = f"{dataset}_{model}"
            estimator = ensembles.make_estimator(model, **best_params(args.best_dir, dataset, model)).fit(X, y)
            values, X_used = explain(estimator, X, label, log)
            np.savez_compressed(args.out / f"shap_{label}.npz", values=values.values, base=values.base_values,
                                rows=X_used.index.astype("int64"))
            name = f"{DATASET_TITLES[dataset]}, {paper.MODEL_NAMES[model]}"

            if (dataset, model) in IMPORTANCE_FIGURES:
                importance_plot(estimator, model, IMPORTANCE_FIGURES[dataset, model], f"Feature importance: {name}", args.out)

            plt.figure()
            shap.summary_plot(values, X_used, show=False, plot_size=(6, 5))
            save(SUMMARY_FIGURES[dataset, model], f"SHAP summary: {name}", args.out)

            for fig_name, d, m, feature, colour in DEPENDENCE_FIGURES:
                if (d, m) == (dataset, model):
                    shap.dependence_plot(feature, values.values, X_used, interaction_index=colour, show=False)
                    save(fig_name, f"SHAP dependence of {feature} (colour: {colour}): {name}", args.out)

            for fig_name, d, m, pair in PDP_FIGURES:
                if (d, m) == (dataset, model):
                    from sklearn.inspection import PartialDependenceDisplay

                    fig, ax = plt.subplots(figsize=(6, 5))
                    PartialDependenceDisplay.from_estimator(estimator, X, [pair], ax=ax)
                    save(fig_name, f"Partial dependence of consumption (kWh) on {pair[0]} and {pair[1]}: {name}", args.out)

            if (dataset, model) == ("aep", "gbm"):
                X_test = test[TOTAL].iloc[:DECISION_PLOT_ROWS]
                test_values = shap.TreeExplainer(estimator)(X_test)
                plt.figure()
                shap.decision_plot(test_values.base_values[0], test_values.values, X_test,
                                   feature_order="hclust", show=False)
                save("fig11", f"SHAP decision plot, first {DECISION_PLOT_ROWS} test hours (Wh): {name}", args.out)

    (args.out / "xai_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
