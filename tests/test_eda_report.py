"""Tests unitaires de la feature eda-report."""

from pathlib import Path

import pandas as pd

from src.data_loading import load_data
from src.eda_report import generate_summary, save_report

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "churn_sample.csv"


def test_summary_contient_les_sections():
    df = load_data(SAMPLE)
    summary = generate_summary(df)
    for section in ("dimensions", "valeurs_manquantes", "statistiques", "distribution_churn"):
        assert section in summary


def test_dimensions_du_sample():
    df = load_data(SAMPLE)
    summary = generate_summary(df)
    assert summary["dimensions"] == {"lignes": 12, "colonnes": 7}


def test_valeurs_manquantes_comptees(tmp_path):
    fichier = tmp_path / "manquants.csv"
    fichier.write_text(
        "customer_id,age,tenure_months,monthly_charges,contract_type,support_calls,churn\n"
        "C001,34,12,65.5,month-to-month,3,1\n"
        "C002,,36,89.9,two-year,0,0\n"
        "C003,23,,45.0,month-to-month,5,1\n",
        encoding="utf-8",
    )
    df = load_data(fichier)
    manquantes = generate_summary(df)["valeurs_manquantes"]
    assert manquantes["age"] == 1
    assert manquantes["tenure_months"] == 1
    assert manquantes["customer_id"] == 0


def test_distribution_churn_du_sample():
    df = load_data(SAMPLE)
    distribution = generate_summary(df)["distribution_churn"]
    assert distribution == {0: 7, 1: 5}


def test_save_report_cree_un_fichier_lisible(tmp_path):
    df = load_data(SAMPLE)
    rapport = tmp_path / "rapports" / "eda.md"
    chemin = save_report(df, rapport)
    assert chemin.exists()
    contenu = chemin.read_text(encoding="utf-8")
    assert "# Rapport EDA" in contenu
    assert "## Distribution du churn" in contenu
