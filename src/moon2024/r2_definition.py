"""Which R² did the paper's Figure 12 use? The usual one, or the squared correlation?

  python -m moon2024.r2_definition [--raw-dir outputs/raw] [--out docs/results/fig12_r2.md]

Refits each model of the raw-column experiment on all 3,289 hours with the
settings saved by the grid search (outputs/raw/best_aep-raw_*.json), as
run_raw_figures does, and scores the in-sample predictions both ways. Light: a
few seconds per model. Models whose library isn't installed are listed as such.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from . import data, ensembles, paper
from .features import TARGET

DATASET = "aep-raw"
# Figure 12(f); the same values as run_raw_figures.PAPER_R2, copied because that module needs shap
PAPER = {"rf": 0.964, "gbm": 0.630, "xgboost": 0.786, "lightgbm": 0.831, "catboost": 0.932}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=Path("outputs/raw"))
    parser.add_argument("--out", type=Path, default=Path("docs/results/fig12_r2.md"))
    args = parser.parse_args()

    train, test = data.load_dataset(DATASET)
    full = pd.concat([train, test])
    columns = data.input_sets(DATASET)["raw"]
    lines = ["## Figure 12: which R²?", "",
             "In-sample predictions (fitted and scored on all 3,289 hours), scored two ways.", "",
             "| Model | Usual R² | Squared correlation | Paper |", "|---|---|---|---|"]
    for model in ensembles.MODELS:
        params = json.loads((args.raw_dir / f"best_{DATASET}_{model}.json").read_text())["params"]["raw"]
        try:
            fitted = ensembles.make_estimator(model, **params).fit(full[columns], full[TARGET]).predict(full[columns])
        except ImportError:
            lines.append(f"| {paper.MODEL_NAMES[model]} | not run (library not installed) | | {PAPER[model]:.3f} |")
            continue
        corr2 = np.corrcoef(full[TARGET], fitted)[0, 1] ** 2
        lines.append(f"| {paper.MODEL_NAMES[model]} | {r2_score(full[TARGET], fitted):.3f} | {corr2:.3f} | "
                     f"{PAPER[model]:.3f} |")
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
