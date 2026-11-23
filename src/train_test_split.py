"""Decoupage train/test (feature train-test-split)."""

from sklearn.model_selection import train_test_split


def split_data(df, target="churn", test_size=0.2, seed=42):
    """Separe le DataFrame en X_train, X_test, y_train, y_test.

    Args:
        df: DataFrame contenant la colonne cible.
        target: Nom de la colonne cible.
        test_size: Proportion du jeu de test.
        seed: Graine aleatoire (resultat reproductible).

    Raises:
        ValueError: si la colonne cible est absente.
    """
    if target not in df.columns:
        raise ValueError(f"Colonne cible {target} absente.")

    X = df.drop(columns=[target])
    y = df[target]
    
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)