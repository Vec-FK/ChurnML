"""Tests unitaires de la feature churn-charts."""

from pathlib import Path

from src.churn_charts import plot_churn_by_contract, plot_tenure_vs_churn
from src.data_loading import load_data

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "churn_sample.csv"


def test_plot_churn_by_contract_cree_un_png_non_vide(tmp_path):
    df = load_data(SAMPLE)
    chemin = tmp_path / "churn_par_contrat.png"

    resultat = plot_churn_by_contract(df, chemin)

    assert resultat == chemin
    assert chemin.exists()
    assert chemin.stat().st_size > 0


def test_plot_tenure_vs_churn_cree_un_png_non_vide(tmp_path):
    df = load_data(SAMPLE)
    chemin = tmp_path / "tenure_vs_churn.png"

    resultat = plot_tenure_vs_churn(df, chemin)

    assert resultat == chemin
    assert chemin.exists()
    assert chemin.stat().st_size > 0
