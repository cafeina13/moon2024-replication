"""The 15 input variables of the article's Table 2, built from the hourly table."""

import numpy as np
import pandas as pd

EXTERNAL = ["Hour_x", "Hour_y", "DOTW_x", "DOTW_y", "Holi", "Temp", "Humi", "WS", "THI", "WCT"]
INTERNAL = ["Cons_1", "Holi_1", "Cons_7", "Holi_7", "Cons_avg"]
TOTAL = EXTERNAL + INTERNAL
INPUT_SETS = {"external": EXTERNAL, "internal": INTERNAL, "total": TOTAL}
TARGET = "Consumption"

DAY = pd.Timedelta(hours=24)
WEEK = 7 * DAY
CONS_AVG_DAYS = 7


def thi(temp, humi):
    """Temperature-humidity index, Eq (5); temp in deg C, humi in %."""
    return 1.8 * temp + 32 - (0.55 - 0.0055 * humi) * (1.8 * temp - 26)


def wct(temp, ws):
    """Wind chill temperature, Eq (6); ws used as given in the data (m/s)."""
    return 13.12 + 0.6215 * temp - 11.37 * ws**0.16 + 0.3965 * temp * ws**0.16


def day_of_hour(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The time each hour starts at, which decides the day it belongs to.

    An hour labelled t covers (t - 1 h, t], so 00:00 belongs to the previous
    day. The authors' table follows this for DOTW and Holi (not for Hour).
    """
    return index - pd.Timedelta(hours=1)


def holiday_indicator(index: pd.DatetimeIndex) -> np.ndarray:
    """1 for weekend hours, 0 otherwise.

    Saturday 00:00 (Friday 23:00-24:00) is 0 and Monday 00:00 is 1.
    The article also mentions public holidays from timeanddate.com, but the
    authors' table marks none (e.g. Ascension, 5 May 2016, is 0).
    """
    return (day_of_hour(index).dayofweek >= 5).astype(int)


def _lag(series: pd.Series, by: pd.Timedelta) -> np.ndarray:
    """Value at t - by; 0 where that time is before the data starts."""
    return series.reindex(series.index - by).fillna(0).to_numpy()


def cons_avg(consumption: pd.Series, holi: pd.Series) -> np.ndarray:
    """Mean consumption at the same hour on the 7 most recent earlier days of the same type.

    Day type is the holiday indicator. Earlier days are scanned back one day
    at a time, and fewer than 7 are averaged when history runs out. Hours in
    the first week (no value one week back) are 0, as in the authors' table.
    """
    values = consumption.to_dict()
    types = holi.to_dict()
    start = consumption.index[0]
    out = np.zeros(len(consumption))
    for i, t in enumerate(consumption.index):
        if t - WEEK < start:
            continue
        same_type = []
        past = t - DAY
        while past >= start and len(same_type) < CONS_AVG_DAYS:
            if past in values and types[past] == types[t]:
                same_type.append(values[past])
            past -= DAY
        out[i] = np.mean(same_type) if same_type else 0.0
    return out


def build_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Hourly table (Temp, Humi, WS, Consumption) -> the 15 inputs plus the target."""
    idx = hourly.index
    hour = idx.hour.to_numpy()
    dotw = ((day_of_hour(idx).dayofweek + 1) % 7).to_numpy()  # Sunday = 0, as in the authors' table
    holi = pd.Series(holiday_indicator(idx), index=idx)
    cons = hourly[TARGET]

    out = pd.DataFrame(index=idx)
    out["Hour_x"] = np.sin(2 * np.pi * hour / 24)
    out["Hour_y"] = np.cos(2 * np.pi * hour / 24)
    out["DOTW_x"] = np.sin(2 * np.pi * dotw / 7)
    out["DOTW_y"] = np.cos(2 * np.pi * dotw / 7)
    out["Holi"] = holi
    out["Temp"] = hourly["Temp"]
    out["Humi"] = hourly["Humi"]
    out["WS"] = hourly["WS"]
    out["THI"] = thi(out["Temp"], out["Humi"])
    out["WCT"] = wct(out["Temp"], out["WS"])
    out["Cons_1"] = _lag(cons, DAY)
    out["Holi_1"] = _lag(holi, DAY).astype(int)
    out["Cons_7"] = _lag(cons, WEEK)
    out["Holi_7"] = _lag(holi, WEEK).astype(int)
    out["Cons_avg"] = cons_avg(cons, holi)
    out[TARGET] = cons
    return out
