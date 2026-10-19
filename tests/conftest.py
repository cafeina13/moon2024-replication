import pytest

from moon2024 import data, features


def _load(loader):
    try:
        return loader()
    except FileNotFoundError as missing:
        pytest.skip(str(missing))


@pytest.fixture(scope="session")
def hourly():
    return data.to_hourly(_load(data.load_raw_aep))


@pytest.fixture(scope="session")
def aep(hourly):
    return features.build_features(hourly)


@pytest.fixture(scope="session")
def authors_aep():
    return _load(data.load_authors_aep)


@pytest.fixture(scope="session")
def authors_urc():
    return _load(data.load_authors_urc)
