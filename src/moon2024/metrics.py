"""The article's error measures, Eqs (8)-(11), all in percent."""

import numpy as np
import pandas as pd


def mape(y, y_hat) -> float:
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    return float(np.mean(np.abs((y - y_hat) / y)) * 100)


def cvrmse(y, y_hat) -> float:
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    return float(np.sqrt(np.mean((y - y_hat) ** 2)) / np.mean(y) * 100)


def nmae(y, y_hat) -> float:
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    return float(np.mean(np.abs(y - y_hat)) / np.mean(y) * 100)


def harmonic_mean(*values: float) -> float:
    """Eq (11): 3 / (1/MAPE + 1/CVRMSE + 1/NMAE)."""
    return len(values) / sum(1 / v for v in values)


def scores(y, y_hat) -> dict:
    m, c, n = mape(y, y_hat), cvrmse(y, y_hat), nmae(y, y_hat)
    return {"MAPE": m, "CVRMSE": c, "NMAE": n, "HM": harmonic_mean(m, c, n)}


def scores_by_day_type(y, y_hat, holi) -> pd.DataFrame:
    """Rows W (weekdays, Holi 0), H (holidays and weekends, Holi 1), T (all).

    The mean in CVRMSE and NMAE is taken within each group, as in Tables 5-10.
    """
    y, y_hat, holi = np.asarray(y, float), np.asarray(y_hat, float), np.asarray(holi)
    groups = {"W": holi == 0, "H": holi == 1, "T": np.ones(len(y), bool)}
    return pd.DataFrame({name: scores(y[mask], y_hat[mask]) for name, mask in groups.items()}).T
