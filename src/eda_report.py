"""Rapport exploratoire des donnees (feature eda-report).

Deuxieme etape du pipeline ChurnML : produire un resume
exploratoire du DataFrame charge par data-loading.
"""

from pathlib import Path

import pandas as pd


def generate_summary(df):
    """Produit le resume exploratoire du DataFrame.

    - dimensions du tableau (lignes, colonnes)
    - valeurs manquantes par colonne
    - statistiques descriptives des colonnes numeriques
    - distribution de la colonne churn (si presente)

    Args:
        df: Le DataFrame a analyser.

    Returns:
        Un dictionnaire contenant les sections du resume.
    """
    numeric = df.select_dtypes(include="number")

    summary = {
        "dimensions": {"lignes": int(len(df)), "colonnes": int(len(df.columns))},
        "valeurs_manquantes": df.isna().sum().to_dict(),
        "statistiques": numeric.describe().round(2).to_dict(),
    }

    if "churn" in df.columns:
        summary["distribution_churn"] = (
            df["churn"].value_counts().sort_index().to_dict()
        )

    return summary


def save_report(df, output_path):
    """Genere un rapport EDA au format Markdown et l'ecrit sur disque.

    Args:
        df: Le DataFrame a analyser.
        output_path: Chemin du fichier .md a ecrire.

    Returns:
        Le chemin (Path) du rapport ecrit.
    """
    summary = generate_summary(df)

    lines = ["# Rapport EDA", ""]
    lines.append(
        f"- Dimensions : {summary['dimensions']['lignes']} lignes "
        f"x {summary['dimensions']['colonnes']} colonnes"
    )

    lines.append("")
    lines.append("## Valeurs manquantes")
    for col, count in summary["valeurs_manquantes"].items():
        lines.append(f"- {col} : {count}")

    lines.append("")
    lines.append("## Statistiques descriptives (colonnes numeriques)")
    for col, stats in summary["statistiques"].items():
        lines.append(
            f"- {col} : moyenne {stats.get('mean')} / min {stats.get('min')} / max {stats.get('max')}"
        )

    if "distribution_churn" in summary:
        lines.append("")
        lines.append("## Distribution du churn")
        total = sum(summary["distribution_churn"].values())
        for value, count in summary["distribution_churn"].items():
            pct = round(100 * count / total, 1)
            lines.append(f"- churn = {value} : {count} ({pct} %)")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return output
