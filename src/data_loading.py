"""Chargement des donnees clients (feature data-loading).

Premiere etape du pipeline ChurnML : lire le CSV des clients
et le transformer en DataFrame pandas utilisable par les
etapes suivantes (eda-report, feature-engineering, ...).
"""

from pathlib import Path

import pandas as pd

# Colonnes obligatoires pour le pipeline ChurnML
REQUIRED_COLUMNS = [
    "customer_id",
    "age",
    "tenure_months",
    "monthly_charges",
    "contract_type",
    "support_calls",
    "churn",
]


def load_data(path, require_target=True):
    """Charge le CSV clients dans un DataFrame pandas.

    - normalise les noms de colonnes (espaces et majuscules supprimes)
    - verifie que les colonnes obligatoires sont presentes
    - exige churn uniquement si require_target vaut True

    Raises:
        FileNotFoundError: si le fichier n'existe pas.
        ValueError: si des colonnes obligatoires manquent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    df = pd.read_csv(path)
    df.columns = [str(col).strip().lower() for col in df.columns]

    required_columns = (
        REQUIRED_COLUMNS
        if require_target
        else [col for col in REQUIRED_COLUMNS if col != "churn"]
    )
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

    return df
