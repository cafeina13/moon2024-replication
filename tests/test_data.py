import numpy as np
import pandas as pd
import pytest

from moon2024 import data, features


def test_table1_statistics(hourly):
    """Article Table 1, AEP column (Wh)."""
    c = hourly["Consumption"]
    assert c.min() == 170.0
    assert c.quantile(0.25) == 300.0
    assert c.median() == 380.0
    assert round(c.mean(), 1) == 586.2
    assert c.quantile(0.75) == 660.0
    assert c.max() == 4000.0


def test_hourly_range(hourly):
    assert hourly.index[0] == pd.Timestamp("2016-01-11 18:00")
    assert hourly.index[-1] == pd.Timestamp("2016-05-27 18:00")
    assert len(hourly) == 3289
    assert (hourly.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()


def test_features_match_authors_table(aep, authors_aep):
    """Our 15 inputs and target equal the authors' published AEP table."""
    assert list(aep.columns) == list(authors_aep.columns)
    assert aep.index.equals(authors_aep.index)
    for column in aep.columns:
        np.testing.assert_allclose(aep[column], authors_aep[column], rtol=1e-6, atol=1e-6, err_msg=column)


def test_aep_split(aep):
    train, test = data.split_aep(aep)
    assert (len(train), len(test)) == (2311, 978)
    assert train.index[-1] == pd.Timestamp("2016-04-17 00:00")
    assert test.index[0] == pd.Timestamp("2016-04-17 01:00")
    assert test.index[-1] == pd.Timestamp("2016-05-27 18:00")
    assert int(test["Holi"].sum()) == 264


def test_urc_table_and_split(authors_urc):
    assert list(authors_urc.columns) == features.TOTAL + [features.TARGET]
    train, test = data.split_urc(authors_urc)
    assert (len(train), len(test)) == (20472, 8760)
    assert train.index[0] == pd.Timestamp("2015-09-01 00:00")
    assert test.index[-1] == pd.Timestamp("2018-12-31 23:00")


def test_urc_table1_statistics(authors_urc):
    """Article Table 1, University Residential Complex column (kWh)."""
    c = authors_urc["Consumption"]
    expected = {"min": 570.2, "q1": 993.2, "median": 1174.0, "mean": 1217.6, "q3": 1392.5, "max": 2588.0}
    actual = {
        "min": c.min(), "q1": c.quantile(0.25), "median": c.median(),
        "mean": c.mean(), "q3": c.quantile(0.75), "max": c.max(),
    }
    for name, value in expected.items():
        assert actual[name] == pytest.approx(value, abs=0.1), name


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_internal_inputs_are_known_a_day_ahead(hourly, aep, seed):
    """Changing consumption after t - 24 h leaves the internal inputs at t unchanged."""
    rng = np.random.default_rng(seed)
    t = hourly.index[rng.integers(200, len(hourly))]
    changed = hourly.copy()
    later = changed.index > t - features.DAY
    changed.loc[later, "Consumption"] = rng.uniform(100, 5000, later.sum())
    rebuilt = features.build_features(changed)
    pd.testing.assert_series_equal(rebuilt.loc[t, features.INTERNAL], aep.loc[t, features.INTERNAL])
