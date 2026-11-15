"""Evaluation du modele (feature evaluation-metrics).

Derniere etape du pipeline ChurnML : mesurer la qualite d'un modele
entraine sur le jeu de test (accuracy, precision, recall, F1).
"""

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate(model, X_test, y_test):
    """Evalue un modele de classification sur le jeu de test.

    Args:
        model: Modele scikit-learn deja entraine (methode predict).
        X_test: DataFrame des variables explicatives du jeu de test.
        y_test: Serie cible reelle (churn : 0 ou 1).

    Returns:
        Un dictionnaire avec les cles accuracy, precision, recall et f1.

    Raises:
        ValueError: si X_test est vide ou si X_test et y_test n'ont
            pas la meme taille.
    """
    if len(X_test) == 0:
        raise ValueError("X_test est vide : evaluation impossible.")

    if len(X_test) != len(y_test):
        raise ValueError(
            f"X_test et y_test n'ont pas la meme taille "
            f"({len(X_test)} contre {len(y_test)})."
        )

    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
