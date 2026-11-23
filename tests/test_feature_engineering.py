"""Tests unitaires de la feature feature-engineering."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_loading import load_data
from src.feature_engineering import add_derived_features, encode_contract, transform

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "churn_sample.csv"


def test_encode_contract_cree_les_colonnes_one_hot():
    df = load_data(SAMPLE)
    encoded = encode_contract(df)
    for colonne in ("contract_month_to_month", "contract_one_year", "contract_two_year"):
        assert colonne in encoded.columns
    assert len(encoded) == len(df)


def test_encode_contract_valeurs_correctes():
    df = load_data(SAMPLE)
    encoded = encode_contract(df)
    ligne1 = encoded.iloc[0]
    assert ligne1["contract_month_to_month"] == 1
    assert ligne1["contract_one_year"] == 0
    assert ligne1["contract_two_year"] == 0


def test_encode_contract_colonne_absente_leve_value_error():
    df = pd.DataFrame({"age": [30]})
    with pytest.raises(ValueError):
        encode_contract(df)


def test_total_charges_calcule():
    df = load_data(SAMPLE)
    transformed = transform(df)
    ligne1 = transformed.iloc[0]
    assert ligne1["total_charges"] == 65.5 * 12


def test_is_new_customer_seuil_six_mois():
    df = pd.DataFrame(
        {
            "monthly_charges": [50.0, 50.0, 50.0],
            "tenure_months": [5, 6, 0],
            "contract_type": ["month-to-month", "one-year", "two-year"],
            "support_calls": [2, 1, 0],
        }
    )
    transformed = transform(df)
    assert list(transformed["is_new_customer"]) == [1, 0, 1]


def test_support_intensity_evite_la_division_par_zero():
    df = pd.DataFrame(
        {
            "monthly_charges": [50.0, 50.0],
            "tenure_months": [0, 4],
            "contract_type": ["month-to-month", "one-year"],
            "support_calls": [3, 5],
        }
    )
    transformed = transform(df)
    assert transformed.iloc[0]["support_intensity"] == 3.0
    assert transformed.iloc[1]["support_intensity"] == 1.25


def test_transform_ne_modifie_pas_l_entree():
    df = load_data(SAMPLE)
    avant = df.copy()
    transform(df)
    assert list(df.columns) == list(avant.columns)
    assert df["tenure_months"].tolist() == avant["tenure_months"].tolist()
    assert df["monthly_charges"].tolist() == avant["monthly_charges"].tolist()


def test_transform_conserve_lignes_cible_et_identifiants():
    df = load_data(SAMPLE)
    transformed = transform(df)
    assert len(transformed) == len(df)
    assert transformed["churn"].tolist() == df["churn"].tolist()
    assert transformed["customer_id"].tolist() == df["customer_id"].tolist()
