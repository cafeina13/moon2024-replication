"""Which epoch would each stopping rule pick, and how good is the test forecast there?

  python -m moon2024.stopping TRACK_DIR [--out docs/results/stopping_aep.md] [--figure FILE]

TRACK_DIR holds the epochs_<dataset>_<model>.csv files of `run_deep --track`: the
training loss, the validation loss and the 24th-hour test scores after every epoch.
Each rule looks only at the training and validation losses, never at the test
scores; the "best test epoch" row is shown only as the best any rule could reach.

Rules (the GL, PQ and UP criteria are from Prechelt, "Early stopping - but when?",
1998). A rule that stops goes back to the epoch with the lowest validation loss so
far, as Keras' restore_best_weights does (the tf-keras used here also restores when training
runs to the end without stopping: the early-stopping run's TCN equals its epoch 99 here).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STRIP = 5  # Prechelt's training strip: progress is measured over the last 5 epochs


def generalisation_loss(val: np.ndarray) -> np.ndarray:
    """GL: how far (in %) the validation loss is above the lowest seen so far."""
    return 100 * (val / np.minimum.accumulate(val) - 1)


def training_progress(loss: np.ndarray) -> np.ndarray:
    """P: how much (in per mille) the mean training loss of the last 5 epochs is above their minimum.

    Large while training still improves quickly, near 0 once it has flattened.
    """
    progress = np.full(len(loss), np.inf)
    for t in range(STRIP - 1, len(loss)):
        strip = loss[t - STRIP + 1:t + 1]
        progress[t] = 1000 * (strip.mean() / strip.min() - 1)
    return progress


def best_so_far(val: np.ndarray, stop: int) -> int:
    """Index of the lowest validation loss up to and including epoch index `stop`."""
    return int(np.argmin(val[:stop + 1]))


def first(condition: np.ndarray) -> int | None:
    hits = np.flatnonzero(condition)
    return int(hits[0]) if len(hits) else None


def choose(epochs: pd.DataFrame) -> dict[str, int]:
    """Epoch index (0-based) picked by each rule."""
    loss, val = epochs["loss"].to_numpy(), epochs["val_loss"].to_numpy()
    last = len(val) - 1
    gl, progress = generalisation_loss(val), training_progress(loss)

    def stop_then_restore(condition):
        stop = first(condition)
        return best_so_far(val, last if stop is None else stop)  # Keras restores the best weights at the end too

    # Patience 10: stop once 10 epochs pass without a new lowest validation loss
    since_best = np.array([t - best_so_far(val, t) for t in range(len(val))])
    # UP: validation loss higher than 5 epochs earlier, at two strip ends in a row
    strip_end = np.arange(len(val)) % STRIP == STRIP - 1
    up = np.zeros(len(val), bool)
    for t in range(3 * STRIP - 1, len(val), STRIP):
        up[t] = val[t] > val[t - STRIP] and val[t - STRIP] > val[t - 2 * STRIP]

    return {
        "Epoch 100 (article)": last,
        "Patience 10 (authors' notebook)": stop_then_restore(since_best >= 10),
        "Lowest validation loss, all 100 epochs": int(np.argmin(val)),
        "Epoch 10": 9,
        "Epoch 25": 24,
        "Epoch 50": 49,
        "GL > 5: validation loss 5% above its best": stop_then_restore(gl > 5),
        "PQ > 1: GL / training progress": stop_then_restore(strip_end & (gl / progress > 1)),
        "UP2: validation loss rose over two strips": stop_then_restore(up),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("track", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/results/stopping_aep.md"))
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()

    runs = {p.stem.split("_", 2)[2]: pd.read_csv(p) for p in sorted(args.track.rglob("epochs_*_*.csv"))}
    rows = []
    for model, epochs in runs.items():
        picks = choose(epochs) | {"Best test epoch (not a rule; uses the test set)": int(epochs["MAPE"].idxmin())}
        for rule, t in picks.items():
            rows.append({"rule": rule, "model": model, "epoch": t + 1, "MAPE": epochs["MAPE"].iloc[t],
                         "CVRMSE": epochs["CVRMSE"].iloc[t]})
    table = pd.DataFrame(rows)
    rules = list(dict.fromkeys(table["rule"]))

    mape = table.pivot(index="rule", columns="model", values="MAPE").loc[rules, list(runs)]
    epoch = table.pivot(index="rule", columns="model", values="epoch").loc[rules, list(runs)]
    summary = table.groupby("rule", sort=False)[["MAPE", "CVRMSE"]].mean()

    lines = ["## Stopping rules on the AEP deep networks", "",
             "Trained 100 epochs on 90% of the training windows (10% held back for validation); the test hours were "
             "scored after every epoch. Each cell: test MAPE (24th hour) at the epoch the rule picks, with that epoch "
             "in brackets. The last two columns are means over the ten networks.", "",
             "| Rule | " + " | ".join(runs) + " | Mean MAPE | Mean CVRMSE |",
             "|---|" + "---|" * (len(runs) + 2)]
    for rule in rules:
        cells = [f"{mape.loc[rule, m]:.1f} ({epoch.loc[rule, m]})" for m in runs]
        lines.append(f"| {rule} | " + " | ".join(cells)
                     + f" | {summary.loc[rule, 'MAPE']:.1f} | {summary.loc[rule, 'CVRMSE']:.1f} |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharex=True)
        for ax, (model, epochs) in zip(axes.flat, runs.items()):
            ax.plot(epochs["epoch"], epochs["MAPE"], color="tab:red", lw=1, label="test MAPE")
            ax.set_ylim(0, min(200, epochs["MAPE"].max() * 1.05))
            ax.set_title(model)
            twin = ax.twinx()
            twin.plot(epochs["epoch"], epochs["loss"], color="tab:gray", lw=1, label="training loss")
            twin.plot(epochs["epoch"], epochs["val_loss"], color="tab:blue", lw=1, label="validation loss")
            twin.set_yscale("log")
            for t, style in ((epoch.loc["Patience 10 (authors' notebook)", model], ":"),
                             (epoch.loc["Lowest validation loss, all 100 epochs", model], "--")):
                ax.axvline(t, color="black", ls=style, lw=0.8)
        axes[0, 0].legend(loc="upper left")
        twin.legend(loc="upper right")
        fig.supxlabel("epoch (dotted: patience 10; dashed: lowest validation loss)")
        fig.tight_layout()
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=120)


if __name__ == "__main__":
    main()
