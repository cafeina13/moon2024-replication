import numpy as np
import pytest

from moon2024 import metrics

Y = [100.0, 200.0, 300.0, 400.0]
Y_HAT = [110.0, 180.0, 300.0, 440.0]


def test_mape():
    # (10/100 + 20/200 + 0 + 40/400) / 4 = 0.075
    assert metrics.mape(Y, Y_HAT) == pytest.approx(7.5)


def test_cvrmse():
    # sqrt((100 + 400 + 0 + 1600) / 4) / 250 = sqrt(525) / 250
    assert metrics.cvrmse(Y, Y_HAT) == pytest.approx(np.sqrt(525) / 250 * 100)


def test_nmae():
    # (10 + 20 + 0 + 40) / 4 / 250 = 0.07
    assert metrics.nmae(Y, Y_HAT) == pytest.approx(7.0)


# (MAPE_T, CVRMSE_T, NMAE_T) from Tables 8, 9, 10 and the harmonic mean from Table 12
PAPER_AEP = {
    "external": {
        "RF": ((45.60, 68.61, 44.68), 50.95), "GBM": ((43.01, 73.75, 44.99), 50.82),
        "XGBoost": ((47.11, 78.43, 47.35), 54.45), "LightGBM": ((46.13, 74.06, 46.75), 53.03),
        "CatBoost": ((45.88, 74.38, 46.79), 52.99),
    },
    "internal": {
        "RF": ((46.70, 74.09, 45.80), 52.87), "GBM": ((41.50, 75.07, 43.52), 49.68),
        "XGBoost": ((52.36, 82.75, 51.37), 59.23), "LightGBM": ((51.60, 79.00, 49.87), 57.59),
        "CatBoost": ((48.79, 77.16, 47.69), 55.12),
    },
    "total": {
        "RF": ((46.14, 66.28, 43.94), 50.40), "GBM": ((40.37, 71.93, 42.64), 48.29),
        "XGBoost": ((52.68, 75.89, 49.32), 57.21), "LightGBM": ((44.39, 69.63, 44.04), 50.34),
        "CatBoost": ((41.35, 68.69, 42.61), 48.23),
    },
}


@pytest.mark.parametrize("inputs", PAPER_AEP)
def test_harmonic_mean_matches_table12(inputs):
    """Table 12 from Tables 8-10. The paper used unrounded values, so allow 0.01."""
    for model, (values, expected) in PAPER_AEP[inputs].items():
        assert metrics.harmonic_mean(*values) == pytest.approx(expected, abs=0.011), model


def test_scores_by_day_type_uses_group_means():
    holi = [0, 0, 1, 1]
    table = metrics.scores_by_day_type(Y, Y_HAT, holi)
    assert list(table.index) == ["W", "H", "T"]
    # Holiday group: errors 0 and 40 over mean 350
    assert table.loc["H", "NMAE"] == pytest.approx(20 / 350 * 100)
    assert table.loc["T", "MAPE"] == pytest.approx(7.5)
