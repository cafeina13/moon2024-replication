"""Figures 2 and 3: Taylor diagrams of the five ensembles on the test part.

  python -m moon2024.taylor RESULTS_DIR [--out docs/figures]

Like the published figures, each diagram shows all 15 models: five ensembles
times three input sets (external, internal, total).
A Taylor diagram places each model by the standard deviation of its predictions
(distance from the origin) and their correlation with the actual values (angle);
the distance to the "Observed" point is the centred RMSE.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import paper
from .compare import DATASET_NAMES

FIGURES = {"urc": ("fig02", "kWh"), "aep": ("fig03", "Wh")}
INPUTS = ["total", "external", "internal"]
# The published figures' colours, in their legend order (models alphabetically, then total, external, internal)
PAPER_COLOURS = {
    "CatBoost (total)": "#E41A1C", "CatBoost (external)": "#377EB8", "CatBoost (internal)": "#4DAF4A",
    "GBM (total)": "#984EA3", "GBM (external)": "#FF7F00", "GBM (internal)": "#FFFF33",
    "LightGBM (total)": "#A65628", "LightGBM (external)": "#F781BF", "LightGBM (internal)": "#999999",
    "RF (total)": "#8DD3C7", "RF (external)": "#B3DE69", "RF (internal)": "#BEBADA",
    "XGBoost (total)": "#FB8072", "XGBoost (external)": "#80B1D3", "XGBoost (internal)": "#FDB462",
}


def stats(results: Path, dataset: str) -> pd.DataFrame:
    rows = {}
    for model, model_name in paper.MODEL_NAMES.items():
        for inputs in INPUTS:
            pred = pd.read_csv(results / f"pred_{dataset}_{model}_{inputs}.csv")
            error = (pred["y_hat"] - pred["y_hat"].mean()) - (pred["y"] - pred["y"].mean())
            rows[f"{model_name} ({inputs})"] = {
                "std": pred["y_hat"].std(ddof=0),
                "corr": np.corrcoef(pred["y"], pred["y_hat"])[0, 1],
                "crmse": np.sqrt(np.mean(error**2)),
            }
    rows["observed"] = {"std": pred["y"].std(ddof=0), "corr": 1.0, "crmse": 0.0}
    return pd.DataFrame(rows).T


def taylor_diagram(table: pd.DataFrame, title: str, unit: str, path: Path) -> None:
    ref = table.loc["observed", "std"]
    limit = 1.5 * max(table["std"].max(), ref)
    fig = plt.figure(figsize=(6.5, 6))
    ax = fig.add_subplot(projection="polar")
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_ylim(0, limit)

    # Correlation labels on the arc
    corrs = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
    ax.set_xticks(np.arccos(corrs))
    ax.set_xticklabels([f"{c:g}" for c in corrs])
    ax.text(np.pi / 4, limit * 1.12, "Correlation", ha="center", va="center", rotation=-45)
    ax.set_ylabel(f"Standard deviation ({unit})", labelpad=28)

    # Centred-RMSE contours around the observed point
    theta, radius = np.meshgrid(np.linspace(0, np.pi / 2, 200), np.linspace(0, limit, 200))
    crmse = np.sqrt(ref**2 + radius**2 - 2 * ref * radius * np.cos(theta))
    contours = ax.contour(theta, radius, crmse, levels=5, colors="0.6", linewidths=0.8, linestyles="--")
    ax.clabel(contours, fmt="%.0f", fontsize=7)

    ax.plot(0, ref, "o", color="#A020F0", markersize=9, label="Observed")
    for label, colour in PAPER_COLOURS.items():
        row = table.loc[label]
        ax.plot(np.arccos(row["corr"]), row["std"], "o", color=colour, markersize=9, markeredgecolor="0.4",
                markeredgewidth=0.4, label=f"{label}, r = {row['corr']:.2f}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.1), fontsize=7)
    ax.set_title(title, fontsize=10, pad=30)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for dataset, (fig_name, unit) in FIGURES.items():
        if not (args.results / f"pred_{dataset}_rf_total.csv").exists():
            print(f"skip {fig_name}: no {dataset} predictions yet")
            continue
        table = stats(args.results, dataset)
        print(fig_name, DATASET_NAMES[dataset], "\n", table.round(3))
        taylor_diagram(table, f"Taylor diagram, test part: {DATASET_NAMES[dataset]}", unit, args.out / f"{fig_name}.png")


if __name__ == "__main__":
    main()
