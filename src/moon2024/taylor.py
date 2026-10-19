"""Figures 2 and 3: Taylor diagrams of the five ensembles on the test part.

  python -m moon2024.taylor RESULTS_DIR [--out docs/figures]

Uses the models trained on all 15 inputs (the article does not say which input
set the diagrams show; all 15 is the configuration it reports as best).
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
COLOURS = ["#3b6ea8", "#d1495b", "#edae49", "#00798c", "#66a182"]


def stats(results: Path, dataset: str) -> pd.DataFrame:
    rows = {}
    for model in paper.MODEL_NAMES:
        pred = pd.read_csv(results / f"pred_{dataset}_{model}_total.csv")
        rows[model] = {
            "std": pred["y_hat"].std(ddof=0),
            "corr": np.corrcoef(pred["y"], pred["y_hat"])[0, 1],
            "crmse": np.sqrt(np.mean(((pred["y_hat"] - pred["y_hat"].mean()) - (pred["y"] - pred["y"].mean())) ** 2)),
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
    corrs = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
    ax.set_xticks(np.arccos(corrs))
    ax.set_xticklabels([f"{c:g}" for c in corrs])
    ax.text(np.pi / 4, limit * 1.12, "Correlation", ha="center", va="center", rotation=-45)
    ax.set_ylabel(f"Standard deviation ({unit})", labelpad=28)

    # Centred-RMSE contours around the observed point
    theta, radius = np.meshgrid(np.linspace(0, np.pi / 2, 200), np.linspace(0, limit, 200))
    crmse = np.sqrt(ref**2 + radius**2 - 2 * ref * radius * np.cos(theta))
    contours = ax.contour(theta, radius, crmse, levels=5, colors="0.6", linewidths=0.8, linestyles="--")
    ax.clabel(contours, fmt="%.0f", fontsize=7)

    ax.plot(0, ref, "k*", markersize=14, label="Observed")
    for colour, (model, row) in zip(COLOURS, table.drop(index="observed").iterrows()):
        ax.plot(np.arccos(row["corr"]), row["std"], "o", color=colour, markersize=9,
                label=f"{paper.MODEL_NAMES[model]} (r = {row['corr']:.2f})")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
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
