"""Tests de la CLI de prediction (feature predict-cli)."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_loading import load_data
from src.feature_engineering import transform
from src.model_persistence import save_model
from src.model_training import train
from src.predict_cli import main, predict

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample" / "churn_sample.csv"
NON_FEATURES = ["customer_id", "contract_type", "churn"]


@pytest.fixture
def trained_model():
    """Entraine un petit modele sur l'echantillon."""
    df = load_data(SAMPLE)
    X = transform(df).drop(columns=NON_FEATURES)
    return train(X, df["churn"])


def test_predict_returns_one_row_per_customer(trained_model):
    df = load_data(SAMPLE)
    result = predict(trained_model, df)

    assert len(result) == len(df)
    assert list(result.columns) == [
        "customer_id",
        "churn_prediction",
        "churn_probability",
    ]
    assert set(result["churn_prediction"]).issubset({0, 1})


def test_main_prints_predictions(trained_model, tmp_path, capsys):
    model_path = tmp_path / "model.joblib"
    save_model(trained_model, model_path)

    main(["--model", str(model_path), "--input", str(SAMPLE)])

    assert "churn_prediction" in capsys.readouterr().out


def test_main_writes_output_csv(trained_model, tmp_path):
    model_path = tmp_path / "model.joblib"
    out_path = tmp_path / "predictions.csv"
    save_model(trained_model, model_path)

    main(["--model", str(model_path), "--input", str(SAMPLE), "--output", str(out_path)])

    assert len(pd.read_csv(out_path)) == len(load_data(SAMPLE))


def test_main_predicts_unlabeled_csv(trained_model, tmp_path):
    model_path = tmp_path / "model.joblib"
    input_path = tmp_path / "unlabeled.csv"
    output_path = tmp_path / "predictions.csv"
    save_model(trained_model, model_path)
    load_data(SAMPLE).drop(columns=["churn"]).to_csv(input_path, index=False)

    main(
        [
            "--model",
            str(model_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
    )

    result = pd.read_csv(output_path)
    assert len(result) == len(pd.read_csv(input_path))
    assert list(result.columns) == [
        "customer_id",
        "churn_prediction",
        "churn_probability",
    ]


def test_main_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        main(["--model", str(tmp_path / "absent.joblib"), "--input", str(SAMPLE)])