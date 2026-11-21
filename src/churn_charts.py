"""Visualisations du churn (feature churn-charts).

Deux graphiques matplotlib sauvegardes en PNG, sans affichage interactif,
pour completer le resume textuel produit par la feature eda-report.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - import apres le choix du backend

CONTRATS = ("month-to-month", "one-year", "two-year")


def _chemin_png(chemin):
    """Cree le dossier parent si besoin et retourne un Path."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    return chemin


def plot_churn_by_contract(df, chemin):
    """Barres du taux de churn par type de contrat.

    Args:
        df: DataFrame avec les colonnes contract_type et churn.
        chemin: chemin du fichier PNG a creer.

    Returns:
        Le chemin du PNG cree.
    """
    chemin = _chemin_png(chemin)
    taux = df.groupby("contract_type")["churn"].mean().mul(100)
    taux = taux.reindex([contrat for contrat in CONTRATS if contrat in taux.index])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(taux.index, taux.values, color="#4C72B0")
    ax.set_title("Taux de churn par type de contrat")
    ax.set_xlabel("Type de contrat")
    ax.set_ylabel("Taux de churn (%)")
    ax.set_ylim(0, 100)
    for index, valeur in enumerate(taux.values):
        ax.text(index, valeur + 1, f"{valeur:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(chemin, dpi=120)
    plt.close(fig)

    return chemin


def plot_tenure_vs_churn(df, chemin):
    """Histogramme superpose de l'anciennete selon le churn.

    Args:
        df: DataFrame avec les colonnes tenure_months et churn.
        chemin: chemin du fichier PNG a creer.

    Returns:
        Le chemin du PNG cree.
    """
    chemin = _chemin_png(chemin)

    fig, ax = plt.subplots(figsize=(7, 4))
    series = (
        (0, "Pas de churn", "#55A868"),
        (1, "Churn", "#C44E52"),
    )
    for valeur, libelle, couleur in series:
        anciennete = df.loc[df["churn"] == valeur, "tenure_months"]
        ax.hist(anciennete, bins=10, alpha=0.6, label=libelle, color=couleur)
    ax.set_title("Anciennete selon le churn")
    ax.set_xlabel("Anciennete (mois)")
    ax.set_ylabel("Nombre de clients")
    ax.legend()
    fig.tight_layout()
    fig.savefig(chemin, dpi=120)
    plt.close(fig)

    return chemin
