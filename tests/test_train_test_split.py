"""Tests unitaires de la feature train-test-split."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_loading import load_data
from src.train_test_split import split_data

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "churn_sample.csv"


def make_df():
    return pd.DataFrame({
        "age": list(range(20, 30)),
        "churn": [0, 1] * 5,
    })


def test_split_sizes():
    X_train, X_test, y_train, y_test = split_data(make_df(), test_size=0.2)
    assert len(X_test) == 2
    assert len(X_train) == 8
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_target_not_in_features():
    X_train, X_test, _, _ = split_data(make_df())
    assert "churn" not in X_train.columns
    assert "churn" not in X_test.columns


def test_split_is_reproducible():
    a = split_data(make_df(), seed=1)[1]
    b = split_data(make_df(), seed=1)[1]
    assert list(a.index) == list(b.index)


def test_missing_target_raises():
    with pytest.raises(ValueError):
        split_data(pd.DataFrame({"age": [1, 2]}))


def test_split_sur_le_sample():
    df = load_data(SAMPLE)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.25)
    assert len(X_train) == 9
    assert len(X_test) == 3
    assert len(y_train) == 9
    assert len(y_test) == 3