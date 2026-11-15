"""Tests unitaires de la feature model-persistence."""

import pandas as pd
import pytest

from src.model_persistence import load_model, save_model
from src.model_training import train


def make_data():
    X = pd.DataFrame({
        "age": [25, 40, 31, 52, 23, 60, 35, 47, 29, 55],
        "tenure_months": [2, 30, 5, 48, 1, 60, 12, 36, 3, 50],
        "monthly_charges": [80, 50, 75, 40, 90, 35, 70, 45, 85, 38],
    })
    y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    return X, y


def test_save_creates_file(tmp_path):
    X, y = make_data()
    path = save_model(train(X, y), tmp_path / "model.joblib")
    assert path.is_file()


def test_save_creates_parent_folders(tmp_path):
    X, y = make_data()
    path = save_model(train(X, y), tmp_path / "models" / "v1" / "model.joblib")
    assert path.is_file()


def test_loaded_model_gives_same_predictions(tmp_path):
    X, y = make_data()
    model = train(X, y)
    path = save_model(model, tmp_path / "model.joblib")
    loaded = load_model(path)
    assert list(loaded.predict(X)) == list(model.predict(X))


def test_save_untrained_object_raises(tmp_path):
    with pytest.raises(ValueError):
        save_model(None, tmp_path / "model.joblib")
    with pytest.raises(ValueError):
        save_model("pas un modele", tmp_path / "model.joblib")


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "absent.joblib")
