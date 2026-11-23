"""Tests unitaires de la feature evaluation-metrics."""

import pandas as pd
import pytest

from src.evaluation_metrics import evaluate


class FakeModel:
    """Faux modele qui renvoie des predictions fixes."""

    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, X):
        return self.predictions


def make_data():
    X = pd.DataFrame({"age": [25, 40, 31, 52], "tenure_months": [2, 30, 5, 48]})
    y = pd.Series([1, 1, 0, 0])
    return X, y


def test_returns_expected_keys():
    X, y = make_data()
    result = evaluate(FakeModel([1, 1, 0, 0]), X, y)
    assert set(result) == {"accuracy", "precision", "recall", "f1"}


def test_perfect_model_scores_one():
    X, y = make_data()
    result = evaluate(FakeModel([1, 1, 0, 0]), X, y)
    assert all(value == 1.0 for value in result.values())


def test_known_values():
    X, y = make_data()
    result = evaluate(FakeModel([1, 0, 0, 0]), X, y)
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(2 / 3)


def test_no_positive_prediction_gives_zero():
    X, y = make_data()
    result = evaluate(FakeModel([0, 0, 0, 0]), X, y)
    assert result["precision"] == 0.0
    assert result["f1"] == 0.0


def test_empty_data_raises():
    with pytest.raises(ValueError):
        evaluate(FakeModel([]), pd.DataFrame({"age": []}), pd.Series([], dtype=int))


def test_size_mismatch_raises():
    X, y = make_data()
    with pytest.raises(ValueError):
        evaluate(FakeModel([1, 0]), X, y.iloc[:2])
