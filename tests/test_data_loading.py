"""Tests unitaires de la feature data-loading."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_loading import REQUIRED_COLUMNS, load_data

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "churn_sample.csv"


def test_load_returns_dataframe():
    df = load_data(SAMPLE)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 12


def test_columns_are_normalized():
    df = load_data(SAMPLE)
    assert list(df.columns) == REQUIRED_COLUMNS


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_data("data/sample/inexistant.csv")


def test_missing_column_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("customer_id,age\nC001,34\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_data(bad)


def test_load_without_target_when_not_required(tmp_path):
    unlabeled = tmp_path / "unlabeled.csv"
    unlabeled.write_text(
        "customer_id,age,tenure_months,monthly_charges,contract_type,support_calls\n"
        "C001,34,12,70,month-to-month,2\n",
        encoding="utf-8",
    )

    df = load_data(unlabeled, require_target=False)

    assert "churn" not in df.columns
