"""Contrôle de qualité des données clients.

Expose check_quality(df), qui retourne un petit rapport (dict) : nombre de
lignes, doublons, valeurs manquantes par colonne, et valeurs invalides pour
les colonnes `churn` et `contract_type`. Le module ne dépend que de pandas.
"""

REQUIRED_COLUMNS = ("churn", "contract_type")
VALID_CONTRACTS = {"month-to-month", "one-year", "two-year"}


def check_quality(df):
    """Retourne un rapport de qualité pour un DataFrame de clients.

    Clés du dictionnaire retourné :
    - n_rows : nombre de lignes ;
    - n_duplicates : nombre de lignes dupliquées ;
    - missing : {colonne: nombre de valeurs manquantes} ;
    - invalid_churn : lignes où churn n'est ni 0 ni 1 (une valeur manquante
      compte comme invalide) ;
    - invalid_contract : lignes où contract_type n'est pas dans VALID_CONTRACTS.

    Lève ValueError si une des colonnes de REQUIRED_COLUMNS est absente.
    """
    absent = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if absent:
        raise ValueError(f"Missing required column(s): {absent}")

    return {
        "n_rows": int(len(df)),
        "n_duplicates": int(df.duplicated().sum()),
        "missing": {col: int(n) for col, n in df.isna().sum().items()},
        "invalid_churn": int((~df["churn"].isin([0, 1])).sum()),
        "invalid_contract": int((~df["contract_type"].isin(VALID_CONTRACTS)).sum()),
    }
