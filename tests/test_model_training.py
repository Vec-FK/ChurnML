"""Tests unitaires de la feature model-training."""

import pandas as pd
import pytest

from src.model_training import train


def make_data():
    X = pd.DataFrame({
        "age": [25, 40, 31, 52, 23, 60, 35, 47, 29, 55],
        "tenure_months": [2, 30, 5, 48, 1, 60, 12, 36, 3, 50],
        "monthly_charges": [80, 50, 75, 40, 90, 35, 70, 45, 85, 38],
    })
    y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    return X, y


def test_train_returns_fitted_model():
    X, y = make_data()
    model = train(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds) <= {0, 1}


def test_train_logistic_regression():
    X, y = make_data()
    model = train(X, y, model_name="logistic_regression")
    assert len(model.predict(X)) == len(y)


def test_train_is_reproducible():
    X, y = make_data()
    a = train(X, y, seed=1).predict_proba(X)
    b = train(X, y, seed=1).predict_proba(X)
    assert (a == b).all()


def test_unknown_model_raises():
    X, y = make_data()
    with pytest.raises(ValueError):
        train(X, y, model_name="inconnu")


def test_empty_data_raises():
    with pytest.raises(ValueError):
        train(pd.DataFrame({"age": []}), pd.Series([], dtype=int))


def test_size_mismatch_raises():
    X, y = make_data()
    with pytest.raises(ValueError):
        train(X, y.iloc[:5])


def test_non_numeric_column_raises():
    X, y = make_data()
    X["contract_type"] = "month-to-month"
    with pytest.raises(ValueError):
        train(X, y)
