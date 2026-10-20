"""Our ensemble results next to the paper's Tables 5-12.

  python -m moon2024.compare RESULTS_DIR [--deep DEEP_DIR] [--out docs/results]

RESULTS_DIR holds the pred_<dataset>_<model>_<inputs>.csv files written by
run_ensembles. Writes one Markdown file per table with our value, the paper's
value and the difference for every cell. With --deep (a folder holding the
deep_<dataset>.json files of run_deep), also writes Tables 13 and 14, scored
both ways: the 24th forecast hour (the article) and every hour of every window
(the authors' notebook).
"""

import argparse
from pathlib import Path

import pandas as pd

from . import metrics, paper
from .features import INPUT_SETS

INPUT_NAMES = {"external": "external inputs", "internal": "internal inputs", "total": "all 15 inputs"}
DATASET_NAMES = {"aep": "Appliances Energy Prediction", "urc": "University Residential Complex"}


def ensemble_scores(results: Path, dataset: str, model: str, inputs: str) -> pd.Series | None:
    path = results / f"pred_{dataset}_{model}_{inputs}.csv"
    if not path.exists():
        return None
    pred = pd.read_csv(path)
    table = metrics.scores_by_day_type(pred["y"], pred["y_hat"], pred["Holi"])
    return pd.Series({f"{m}_{g}": table.loc[g, m] for m in ("MAPE", "CVRMSE", "NMAE") for g in "WHT"} | {"HM_T": table.loc["T", "HM"]})


def _cell(ours: float | None, theirs: float) -> str:
    if ours is None:
        return f"– / {theirs:.2f}"
    difference = round(ours, 2) - theirs
    return f"{ours:.2f} / {theirs:.2f} ({difference + 0.0:+.2f})"


def ensemble_table(results: Path, dataset: str, inputs: str) -> str:
    number, rows = paper.ENSEMBLE_TABLES[(dataset, inputs)]
    header = "| Model | " + " | ".join(c.replace("_", " ") for c in paper.COLUMNS) + " |"
    lines = [
        f"## Table {number}: {DATASET_NAMES[dataset]}, {INPUT_NAMES[inputs]}",
        "",
        "Each cell: ours / paper (difference), in %.",
        "",
        header,
        "|" + "---|" * (len(paper.COLUMNS) + 1),
    ]
    for model, theirs in rows.items():
        ours = ensemble_scores(results, dataset, model, inputs)
        cells = [_cell(None if ours is None else ours[c], t) for c, t in zip(paper.COLUMNS, theirs)]
        lines.append(f"| {paper.MODEL_NAMES[model]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def harmonic_mean_table(results: Path, dataset: str) -> str:
    number, rows = paper.HARMONIC_MEAN_TABLES[dataset]
    lines = [
        f"## Table {number}: harmonic mean, {DATASET_NAMES[dataset]}",
        "",
        "Each cell: ours / paper (difference), in %.",
        "",
        "| Model | External | Internal | Total |",
        "|---|---|---|---|",
    ]
    for model, theirs in rows.items():
        cells = []
        for inputs, t in zip(INPUT_SETS, theirs):
            ours = ensemble_scores(results, dataset, model, inputs)
            cells.append(_cell(None if ours is None else ours["HM_T"], t))
        lines.append(f"| {paper.MODEL_NAMES[model]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def deep_table(results: Path, deep: Path, dataset: str) -> str:
    import json

    number, rows = paper.DEEP_TABLES[dataset]
    models = {}
    for path in deep.rglob(f"deep_{dataset}.json"):
        models.update(json.loads(path.read_text())["models"])
    lines = [
        f"## Table {number}: deep networks, {DATASET_NAMES[dataset]}",
        "",
        "Each metric: ours scored on the 24th forecast hour (the article) / ours scored on every hour of every "
        "test window (the authors' notebook) / paper, in %. The last row is the best ensemble (all 15 inputs).",
        "",
        "| Model | MAPE | CVRMSE | NMAE |",
        "|---|---|---|---|",
    ]
    for model, theirs in rows.items():
        if model in models:
            hour24, notebook = models[model]["hour24"], models[model]["notebook_scoring"]
            cells = [f"{hour24[m]:.2f} / {notebook[m]:.2f} / {t:.2f}" for m, t in zip(("MAPE", "CVRMSE", "NMAE"), theirs)]
        elif model in paper.MODEL_NAMES:
            ours = ensemble_scores(results, dataset, model, "total")
            cells = [_cell(None if ours is None else ours[f"{m}_T"], t) for m, t in zip(("MAPE", "CVRMSE", "NMAE"), theirs)]
        else:
            cells = [f"– / – / {t:.2f}" for t in theirs]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--deep", type=Path, help="folder with the deep_<dataset>.json files of run_deep")
    parser.add_argument("--out", type=Path, default=Path("docs/results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for (dataset, inputs), (number, _) in paper.ENSEMBLE_TABLES.items():
        (args.out / f"table{number:02d}.md").write_text(ensemble_table(args.results, dataset, inputs), encoding="utf-8")
    for dataset, (number, _) in paper.HARMONIC_MEAN_TABLES.items():
        (args.out / f"table{number:02d}.md").write_text(harmonic_mean_table(args.results, dataset), encoding="utf-8")
    if args.deep:
        for dataset, (number, _) in paper.DEEP_TABLES.items():
            (args.out / f"table{number:02d}.md").write_text(deep_table(args.results, args.deep, dataset), encoding="utf-8")
    print(f"wrote tables to {args.out}")


if __name__ == "__main__":
    main()
