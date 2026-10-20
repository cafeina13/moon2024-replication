"""Are the differences between models larger than chance? (Beyond the paper, which reports no tests.)

  python -m moon2024.significance [--outputs outputs] [--out docs/results/significance.md] [--draws 2000]

Two checks on the saved test-set predictions, both allowing for errors in
neighbouring hours being correlated:

- Diebold-Mariano test: for two models, the per-hour difference in loss, with a
  Newey-West variance (Bartlett weights, 24 lags: day-ahead forecasts) and
  Harvey, Leybourne and Newbold's small-sample factor and t distribution. In
  brackets: their own form (equal weights over 23 lags), to show how much the
  variance choice matters. Losses: absolute percentage
  error (the basis of MAPE), squared error (CVRMSE) and absolute error (NMAE).
  Two-sided p-values, not corrected for the several comparisons.
- Day-block bootstrap: whole test days resampled with replacement; in each draw
  every model's harmonic mean is recomputed. Reported: how often each model ranks
  first, and a 95% interval for its harmonic mean minus the best model's.

Comparisons: the five ensembles with all 15 inputs; the best ensemble's three
input sets; the best ensemble against the best deep network (24th-hour scoring).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import metrics, paper
from .compare import DATASET_NAMES

LAG = 24
LOSSES = {
    "APE": lambda y, f: np.abs(y - f) / y,
    "SE": lambda y, f: (y - f) ** 2,
    "AE": lambda y, f: np.abs(y - f),
}
DEEP = {"lstm", "bilstm", "gru", "bigru", "1d-cnn", "tcn", "lstm-tcn", "bilstm-tcn", "gru-tcn", "bigru-tcn"}


def diebold_mariano(d: np.ndarray, lag: int = LAG) -> float:
    """Two-sided p-value that the mean loss difference d is zero."""
    n = len(d)
    u = d - d.mean()
    variance = u @ u / n
    for k in range(1, lag + 1):
        variance += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n  # Bartlett weights keep it positive
    statistic = d.mean() / np.sqrt(variance / n)
    statistic *= np.sqrt((n + 1 - 2 * lag + lag * (lag - 1) / n) / n)
    return float(2 * stats.t.sf(abs(statistic), df=n - 1))


def diebold_mariano_hln(d: np.ndarray, horizon: int = LAG) -> float:
    """The textbook Harvey-Leybourne-Newbold form: equal weights over horizon - 1 lags (variance can go negative)."""
    n = len(d)
    u = d - d.mean()
    variance = u @ u / n + 2 * sum((u[k:] @ u[:-k]) / n for k in range(1, horizon))
    if variance <= 0:
        return float("nan")
    statistic = d.mean() / np.sqrt(variance / n)
    statistic *= np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    return float(2 * stats.t.sf(abs(statistic), df=n - 1))


def size_check(runs: int = 400, n: int = 978, phi: float = 0.8, seed: int = 0) -> float:
    """Share of p < 0.05 when there is no real difference but errors are strongly correlated (AR(1))."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(runs):
        e = rng.normal(size=n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + e[t]
        hits += diebold_mariano(x) < 0.05
    return hits / runs


def harmonic_mean(y: np.ndarray, f: np.ndarray) -> float:
    return metrics.scores(y, f)["HM"]


def bootstrap(y: np.ndarray, preds: dict[str, np.ndarray], days: np.ndarray, draws: int, seed: int = 42):
    """Rank-1 share and HM-difference interval per model, resampling whole days."""
    rng = np.random.default_rng(seed)
    unique = np.unique(days)
    rows_of = {d: np.flatnonzero(days == d) for d in unique}
    hm = {m: np.empty(draws) for m in preds}
    for b in range(draws):
        rows = np.concatenate([rows_of[d] for d in rng.choice(unique, len(unique))])
        for m, f in preds.items():
            hm[m][b] = harmonic_mean(y[rows], f[rows])
    table = pd.DataFrame(hm)
    best = min(preds, key=lambda m: harmonic_mean(y, preds[m]))
    first = table.idxmin(axis=1).value_counts(normalize=True)
    diff = table.sub(table[best], axis=0)
    return best, {m: (first.get(m, 0.0), *np.percentile(diff[m], [2.5, 97.5])) for m in preds}


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    return frame


def compare(name: str, frames: dict[str, pd.DataFrame], draws: int, end_labelled: bool) -> list[str]:
    index = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    y = frames[next(iter(frames))].loc[index, "y"].to_numpy(float)
    preds = {m: f.loc[index, "y_hat"].to_numpy(float) for m, f in frames.items()}
    # AEP hours are labelled by their end (00:00 closes the previous day), dormitory hours by their start
    days = (pd.DatetimeIndex(index) - pd.Timedelta(hours=int(end_labelled))).normalize().to_numpy()
    best, boot = bootstrap(y, preds, days, draws)
    lines = [f"### {name}", "",
             f"{len(index)} test hours, {len(np.unique(days))} days. Best by harmonic mean: **{best}**. p-values: "
             "Diebold-Mariano against the best model, per loss.", "",
             "| Model | HM | p (APE) | p (SE) | p (AE) | Ranks first | HM minus best, 95% interval |",
             "|---|---|---|---|---|---|---|"]
    for m, f in sorted(preds.items(), key=lambda kv: harmonic_mean(y, kv[1])):
        if m == best:
            ps = ["–"] * 3
        else:
            ps = [f"{diebold_mariano(d):.3f} ({diebold_mariano_hln(d):.3f})"
                  for d in (loss(y, f) - loss(y, preds[best]) for loss in LOSSES.values())]
        share, low, high = boot[m]
        interval = "–" if m == best else f"{low:+.2f} to {high:+.2f}"
        lines.append(f"| {m} | {harmonic_mean(y, f):.2f} | {' | '.join(ps)} | {share:.0%} | {interval} |")
    return lines + [""]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--out", type=Path, default=Path("docs/results/significance.md"))
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    fixed = args.outputs / "fixed"

    lines = ["## Significance of the differences (beyond the paper)", "",
             "The paper ranks models by their scores without tests. Here: Diebold-Mariano tests (Newey-West variance, "
             f"{LAG} lags, two-sided, not corrected for multiple comparisons) and a day-block bootstrap with "
             f"{args.draws} draws. APE, SE, AE: absolute percentage, squared and absolute error, the losses behind MAPE, "
             "CVRMSE and NMAE. \"Ranks first\": share of bootstrap draws in which the model has the lowest harmonic mean. "
             "p-values in brackets: the textbook Harvey-Leybourne-Newbold variance (equal weights, 23 lags). "
             f"In a simulation with strongly correlated errors (AR(1), coefficient 0.8, 978 hours, no real difference; "
             f"`size_check`) the main test gave p < 0.05 in {size_check():.1%} of 400 runs instead of 5%, so p-values "
             "near 0.05 are weak evidence. The reference model is the best on this same test set, which also favours "
             "finding a difference.",
             ""]
    for dataset in ("urc", "aep"):
        lines += [f"## {DATASET_NAMES[dataset]}", ""]
        ensembles = {paper.MODEL_NAMES[m]: load(fixed / f"pred_{dataset}_{m}_total.csv") for m in paper.MODEL_NAMES}
        lines += compare(f"Ensembles, all 15 inputs (Table {7 if dataset == 'urc' else 10})", ensembles, args.draws, dataset == "aep")

        y_total = {m: metrics.scores(f["y"], f["y_hat"])["HM"] for m, f in ensembles.items()}
        best = min(y_total, key=y_total.get)
        key = {v: k for k, v in paper.MODEL_NAMES.items()}[best]
        sets = {f"{best}, {s}": load(fixed / f"pred_{dataset}_{key}_{s}.csv") for s in ("total", "internal", "external")}
        lines += compare(f"Input sets for {best}", sets, args.draws, dataset == "aep")

        deep = {}
        for path in sorted((args.outputs / "deep").rglob(f"pred_deep_{dataset}_*.csv")):
            deep[path.stem.split("_", 3)[3]] = load(path)
        deep_hm = {m: metrics.scores(f["y"], f["y_hat"])["HM"] for m, f in deep.items()}
        net = min(deep_hm, key=deep_hm.get)
        lines += compare(f"Best ensemble against the best deep network ({net}, 24th hour)",
                         {best: ensembles[best], net: deep[net]}, args.draws, dataset == "aep")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
