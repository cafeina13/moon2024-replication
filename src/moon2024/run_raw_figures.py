"""Figures 12 and 13: the five ensembles on the original AEP columns, run on Kaggle.

  python -m moon2024.run_raw_figures [--best-dir /kaggle/input] [--out /kaggle/working]

The article's Figure 12 reports R² of 0.63-0.96. Those values are reached only
when a model is scored on the hours it was trained on, so each tuned model is
refitted on all hours and scored on them (in-sample, as in the paper), and the
held-out R² on the test part (from run_ensembles finalize) is reported next to
it. Figure 13 is the random forest's SHAP summary on the entire dataset.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import r2_score

from . import data, ensembles, paper
from .features import TARGET
from .run_xai import explain, save

PAPER_R2 = {"rf": 0.964, "gbm": 0.630, "xgboost": 0.786, "lightgbm": 0.831, "catboost": 0.932}
DATASET = data.RAW_DATASET


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--best-dir", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train, test = data.load_dataset(DATASET)
    full = pd.concat([train, test])
    columns = data.input_sets(DATASET)["raw"]
    r2 = {}
    fig, axes = plt.subplots(3, 2, figsize=(9, 12))
    for ax, model in zip(axes.flat, ensembles.MODELS):
        (best,) = args.best_dir.rglob(f"best_{DATASET}_{model}.json")
        params = json.loads(best.read_text())["params"]["raw"]
        estimator = ensembles.make_estimator(model, **params).fit(full[columns], full[TARGET])
        fitted = estimator.predict(full[columns])
        (pred_file,) = args.best_dir.rglob(f"pred_{DATASET}_{model}_raw.csv")
        held_out = pd.read_csv(pred_file)
        r2[model] = {
            "in_sample": r2_score(full[TARGET], fitted),
            "held_out": r2_score(held_out["y"], held_out["y_hat"]),
            "paper": PAPER_R2[model],
        }
        ax.scatter(full[TARGET], fitted, s=4, color="black", alpha=0.5)
        slope, intercept = np.polyfit(full[TARGET], fitted, 1)
        xs = np.array([0, full[TARGET].max()])
        ax.plot(xs, slope * xs + intercept, color="blue")
        ax.set_title(f"{paper.MODEL_NAMES[model]} (in-sample R² = {r2[model]['in_sample']:.3f})", fontsize=9)
        ax.set_xlabel("Actual load (Wh)")
        ax.set_ylabel("Estimated load (Wh)")
        if model == "rf":
            rf = estimator

    ax = axes.flat[-1]
    names = [paper.MODEL_NAMES[m] for m in ensembles.MODELS]
    positions = np.arange(len(names))
    for offset, key, colour, label in [(-0.27, "paper", "0.6", "Paper"), (0, "in_sample", "#3b6ea8", "Ours, in-sample"),
                                       (0.27, "held_out", "#d1495b", "Ours, held-out test")]:
        values = [r2[m][key] for m in ensembles.MODELS]
        ax.bar(positions + offset, values, width=0.27, color=colour, label=label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(positions, names, rotation=30)
    ax.set_ylabel("R²")
    ax.legend(fontsize=7)
    save("fig12", "Hourly AEP original columns: estimated against actual load (all hours)", args.out)

    log = {}
    values, X_used = explain(rf, full[columns], "aep-raw_rf", log)
    plt.figure()
    shap.summary_plot(values, X_used, show=False, plot_size=(6, 8), max_display=len(columns))
    save("fig13", "SHAP summary: RF on the hourly AEP original columns (all hours)", args.out)

    (args.out / "raw_figures.json").write_text(json.dumps({"r2": r2, "shap": log}, indent=2))
    print(json.dumps(r2, indent=2))


if __name__ == "__main__":
    main()
