"""Tests unitaires de la feature model-training."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_loading import load_data
from src.feature_engineering import transform
from src.model_training import train
from src.train_test_split import split_data

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "churn_sample.csv"


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


def test_pipeline_transform_split_train_sur_le_sample():
    df = transform(load_data(SAMPLE))
    df = df.drop(columns=["customer_id", "contract_type"])
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.25)
    model = train(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(preds) <= {0, 1}


def test_train_refuse_les_colonnes_texte_de_transform():
    df = transform(load_data(SAMPLE))
    X = df.drop(columns=["churn"])
    with pytest.raises(ValueError):
        train(X, df["churn"])
