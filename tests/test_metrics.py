import numpy as np
import pytest

from moon2024 import metrics, paper

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


@pytest.mark.parametrize("dataset", paper.HARMONIC_MEAN_TABLES)
def test_harmonic_mean_matches_paper(dataset):
    """Tables 11 and 12 from the T columns of Tables 5-10.

    The paper used unrounded values, so recomputing from the rounded table values may differ by 0.01.
    """
    _, rows = paper.HARMONIC_MEAN_TABLES[dataset]
    for model, expected in rows.items():
        for inputs, hm in zip(["external", "internal", "total"], expected):
            _, table = paper.ENSEMBLE_TABLES[(dataset, inputs)]
            t_values = [table[model][paper.COLUMNS.index(c)] for c in ("MAPE_T", "CVRMSE_T", "NMAE_T")]
            assert metrics.harmonic_mean(*t_values) == pytest.approx(hm, abs=0.011), (model, inputs)


def test_scores_by_day_type_uses_group_means():
    holi = [0, 0, 1, 1]
    table = metrics.scores_by_day_type(Y, Y_HAT, holi)
    assert list(table.index) == ["W", "H", "T"]
    # Holiday group: errors 0 and 40 over mean 350
    assert table.loc["H", "NMAE"] == pytest.approx(20 / 350 * 100)
    assert table.loc["T", "MAPE"] == pytest.approx(7.5)
