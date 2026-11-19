from pathlib import Path

import pandas as pd
import pytest

from clean import clean

MESSY = Path(__file__).resolve().parents[2] / "data" / "housing_messy.csv"
HOODS = {"Leopoldstadt", "Neubau", "Favoriten", "Donaustadt", "Josefstadt", "Floridsdorf"}


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(MESSY)


@pytest.fixture(scope="module")
def out(raw):
    return clean(raw)


def test_does_not_modify_input(raw):
    before = raw.copy(deep=True)
    clean(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_no_duplicate_ids(out):
    assert out["id"].is_unique


def test_neighborhoods_are_the_six_real_ones(out):
    assert set(out["neighborhood"]) == HOODS


def test_balcony_is_boolean(out):
    assert out["has_balcony"].dtype == bool
    assert 0.3 < out["has_balcony"].mean() < 0.6  # about 45% have one


def test_price_is_numeric_and_parsed_correctly(out):
    assert pd.api.types.is_numeric_dtype(out["price_eur"])
    # the dot trap: a naive 'strip every non-digit' turns 397000.0 into 3970000
    assert out["price_eur"].max() < 2_000_000
    assert out["price_eur"].min() > 20_000


def test_dates_are_datetimes_in_the_right_range(out):
    assert pd.api.types.is_datetime64_any_dtype(out["listed_date"])
    assert out["listed_date"].min() >= pd.Timestamp("2023-01-01")
    assert out["listed_date"].max() <= pd.Timestamp("2024-12-31")


def test_no_impossible_values(out):
    assert out["sqm"].between(15, 250).all()
    assert out["year_built"].between(1850, 2024).all()


def test_no_missing_values_and_enough_rows(out):
    assert out.isna().sum().sum() == 0
    assert 500 <= len(out) <= 600
