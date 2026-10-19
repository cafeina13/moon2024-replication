"""Loading the AEP raw data and the authors' prebuilt feature tables.

Article: "we resampled the AEP data to hourly intervals" (no further detail).
The conventions below are the ones that reproduce Table 1 and the authors'
published feature table exactly; see docs/replication_notes.md.
"""

import pandas as pd

from .paths import AUTHORS_AEP, AUTHORS_URC, RAW_AEP, find_file

READINGS_PER_HOUR = 6
WEATHER = {"T_out": "Temp", "RH_out": "Humi", "Windspeed": "WS"}

# Article: training data runs to April 16, 2016 and testing starts on April 17.
# The hour labelled 2016-04-17 00:00 covers 23:00-24:00 on April 16, so it is
# the last training hour (as in the authors' code: the first 2,311 rows).
AEP_TRAIN_END = pd.Timestamp("2016-04-17 00:00")

# Article: training Sept 1, 2015 - Dec 31, 2017; testing the whole of 2018.
URC_TEST_START = pd.Timestamp("2018-01-01 00:00")


def load_raw_aep() -> pd.DataFrame:
    """The UCI 10-minute readings, indexed by timestamp."""
    raw = pd.read_csv(find_file(RAW_AEP), parse_dates=["date"])
    return raw.set_index("date").sort_index()


def to_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """Hourly table labelled by the end of each hour.

    Consumption is the sum of the six readings in (t - 1 h, t]; incomplete
    hours are dropped (only the lone 17:00 reading on 11 January). Weather is
    the reading taken at the hour mark t, which is what the authors' table holds.
    """
    hour_end = raw.index.ceil("h")
    groups = raw["Appliances"].groupby(hour_end)
    hourly = pd.DataFrame({"Consumption": groups.sum().astype(float)})
    hourly = hourly[groups.size() == READINGS_PER_HOUR]

    on_the_hour = raw[raw.index.minute == 0][list(WEATHER)].rename(columns=WEATHER)
    hourly = hourly.join(on_the_hour, how="left")
    hourly.index.name = "Date"
    return hourly[list(WEATHER.values()) + ["Consumption"]]


def to_hourly_raw_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """The original AEP columns, hourly, for the experiment of Figures 12 and 13.

    The article only says the raw data was "sampled hourly" and points to
    Candanedo et al. (2017) for the variables. Energy columns (Appliances,
    lights) are summed over (t - 1 h, t] like the target; every sensor and
    weather column is the reading at t, like the weather in the main table.
    The inputs are the 27 shown in the article's Figure 13: lights, T1-T9,
    RH_1-RH_9, the six weather columns, Day_of_week and WeekStatus. Candanedo
    et al.'s NSM (seconds since midnight) and the random variables rv1, rv2
    are not in the figure, so they are left out. The day is taken from the
    hour's start; WeekStatus is 1 on weekdays and 0 on weekends (the coding in
    the article's text on Figure 13).
    """
    hour_end = raw.index.ceil("h")
    complete = raw.groupby(hour_end).size() == READINGS_PER_HOUR
    energy = raw[["Appliances", "lights"]].groupby(hour_end).sum()[complete].astype(float)
    sensors = raw[raw.index.minute == 0].drop(columns=["Appliances", "lights", "rv1", "rv2"])
    table = energy.join(sensors, how="left")

    day = table.index - pd.Timedelta(hours=1)
    table["WeekStatus"] = (day.dayofweek < 5).astype(int)
    table["Day_of_week"] = day.dayofweek
    table.index.name = "Date"
    return table.rename(columns={"Appliances": "Consumption"})


def load_authors_aep() -> pd.DataFrame:
    """The authors' AEP feature table (github.com/sodayeong/PLOS-ONE_Github)."""
    table = pd.read_csv(find_file(AUTHORS_AEP), parse_dates=["Date"], date_format="%Y-%m-%d %H:%M")
    return table.set_index("Date")


def load_authors_urc() -> pd.DataFrame:
    """The authors' University Residential Complex feature table.

    Only this prebuilt table is public; the raw KEPCO and KMA data are not.
    Its Hour column runs 0-23 and Year is two-digit.
    """
    table = pd.read_csv(find_file(AUTHORS_URC))
    stamp = pd.to_datetime(
        dict(year=table["Year"] + 2000, month=table["Month"], day=table["Day"], hour=table["Hour"])
    )
    return table.drop(columns=["Year", "Month", "Day", "Hour", "Weekday"]).set_index(stamp.rename("Date"))


def split(table: pd.DataFrame, *, train_end: pd.Timestamp | None = None, test_start: pd.Timestamp | None = None):
    """Chronological train/test split; give exactly one boundary."""
    if (train_end is None) == (test_start is None):
        raise ValueError("give exactly one of train_end or test_start")
    is_train = table.index <= train_end if train_end is not None else table.index < test_start
    return table[is_train], table[~is_train]


def split_aep(table: pd.DataFrame):
    return split(table, train_end=AEP_TRAIN_END)


def split_urc(table: pd.DataFrame):
    return split(table, test_start=URC_TEST_START)


DATASETS = ["aep", "urc"]
RAW_DATASET = "aep-raw"  # Figures 12 and 13


def load_dataset(name: str):
    """(train, test) feature tables. AEP is rebuilt from the raw data; URC is the authors' table."""
    from .features import build_features

    if name == "aep":
        return split_aep(build_features(to_hourly(load_raw_aep())))
    if name == "urc":
        return split_urc(load_authors_urc())
    if name == RAW_DATASET:
        return split_aep(to_hourly_raw_columns(load_raw_aep()))
    raise ValueError(f"unknown dataset {name!r}")


def input_sets(name: str) -> dict[str, list[str]]:
    """The input sets a dataset is trained on: external, internal and total, or all raw columns."""
    from .features import INPUT_SETS, TARGET

    if name == RAW_DATASET:
        train, _ = load_dataset(name)
        return {"raw": [c for c in train.columns if c != TARGET]}
    return INPUT_SETS
