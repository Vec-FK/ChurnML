"""Ingenierie de variables (feature feature-engineering).

Troisieme etape du pipeline ChurnML : transformer les variables
brutes issues de data-loading en variables exploitables par le
modele (encodage one-hot, variables derivees).
"""

import pandas as pd

# Categories possibles de contract_type
CONTRACT_VALUES = ["month-to-month", "one-year", "two-year"]


def encode_contract(df):
    """Encode contract_type en one-hot (une colonne binaire par categorie).

    Colonnes creees : contract_month_to_month, contract_one_year,
    contract_two_year. La colonne contract_type d'origine est conservee
    (le modele final choisira les colonnes a utiliser).

    Args:
        df: Le DataFrame a encoder.

    Returns:
        Une copie du DataFrame avec les colonnes one-hot ajoutees.

    Raises:
        ValueError: si la colonne contract_type est absente.
    """
    if "contract_type" not in df.columns:
        raise ValueError("Colonne contract_type absente : encodage impossible.")

    result = df.copy()
    normalisees = result["contract_type"].str.strip().str.lower()

    for value in CONTRACT_VALUES:
        colonne = f"contract_{value.replace('-', '_')}"
        result[colonne] = (normalisees == value).astype(int)

    return result


def add_derived_features(df):
    """Ajoute les variables derivees calculees a partir des colonnes existantes.

    - total_charges : monthly_charges x tenure_months (depense cumulee)
    - is_new_customer : 1 si moins de 6 mois d'anciennete
    - support_intensity : support_calls par mois d'anciennete

    Args:
        df: Le DataFrame a enrichir.

    Returns:
        Une copie du DataFrame avec les colonnes derivees ajoutees.

    Raises:
        ValueError: si une colonne source est absente.
    """
    required = ["monthly_charges", "tenure_months", "support_calls"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes sources manquantes : {missing}")

    result = df.copy()

    result["total_charges"] = result["monthly_charges"] * result["tenure_months"]

    result["is_new_customer"] = (result["tenure_months"] < 6).astype(int)

    # Protection contre la division par zero : un tenure de 0 mois est
    # ramene a 1 mois pour le calcul du ratio.
    tenure_safe = result["tenure_months"].replace(0, 1)
    result["support_intensity"] = (result["support_calls"] / tenure_safe).round(2)

    return result


def transform(df):
    """Applique le pipeline complet d'ingenierie de variables.

    1. Encodage one-hot de contract_type
    2. Ajout des variables derivees

    Le DataFrame d'entree n'est jamais modifie (copies internes).

    Args:
        df: Le DataFrame brut issu de data-loading.

    Returns:
        Le DataFrame transforme.
    """
    result = encode_contract(df)
    result = add_derived_features(result)
    return result
