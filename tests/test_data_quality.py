from pathlib import Path

import pandas as pd

from src.data_quality import check_quality

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "churn_sample.csv"


def make_df():
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 2, 4],
            "churn": [0, 1, 1, 3],
            "contract_type": ["month-to-month", "one-year", "one-year", "weekly"],
            "monthly_charges": [10.0, None, None, 30.0],
        }
    )


def test_report_keys_and_counts():
    result = check_quality(make_df())
    assert result == {
        "n_rows": 4,
        "n_duplicates": 1,
        "missing": {
            "customer_id": 0,
            "churn": 0,
            "contract_type": 0,
            "monthly_charges": 2,
        },
        "invalid_churn": 1,
        "invalid_contract": 1,
    }


def test_clean_dataframe_has_no_issue():
    df = pd.DataFrame({"churn": [0, 1], "contract_type": ["one-year", "two-year"]})
    result = check_quality(df)
    assert result["n_duplicates"] == 0
    assert result["invalid_churn"] == 0
    assert result["invalid_contract"] == 0
    assert sum(result["missing"].values()) == 0


def test_missing_churn_counts_as_invalid():
    df = pd.DataFrame({"churn": [0, None], "contract_type": ["one-year", "one-year"]})
    assert check_quality(df)["invalid_churn"] == 1


def test_empty_dataframe():
    df = pd.DataFrame({"churn": [], "contract_type": []})
    result = check_quality(df)
    assert result["n_rows"] == 0
    assert result["invalid_churn"] == 0
    assert result["invalid_contract"] == 0


def test_missing_columns_count_all_rows_invalid():
    result = check_quality(pd.DataFrame({"customer_id": [1, 2, 3]}))
    assert result["invalid_churn"] == 3
    assert result["invalid_contract"] == 3


def test_sample_file_is_clean():
    df = pd.read_csv(SAMPLE)
    result = check_quality(df)
    assert result["n_rows"] == 12
    assert result["n_duplicates"] == 0
    assert result["invalid_churn"] == 0
    assert result["invalid_contract"] == 0
    assert set(result["missing"]) == set(df.columns)
    assert all(n == 0 for n in result["missing"].values())


def test_sample_file_with_injected_errors():
    df = pd.read_csv(SAMPLE)
    df.loc[0, "churn"] = 5                      # churn invalide
    df.loc[1, "contract_type"] = "weekly"       # contrat invalide
    df.loc[2, "monthly_charges"] = None         # valeur manquante
    df = pd.concat([df, df.iloc[[3]]], ignore_index=True)  # ligne dupliquée
    result = check_quality(df)
    assert result["n_rows"] == 13
    assert result["n_duplicates"] == 1
    assert result["invalid_churn"] == 1
    assert result["invalid_contract"] == 1
    assert result["missing"]["monthly_charges"] == 1