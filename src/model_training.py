"""Entrainement du modele (feature model-training).

Cinquieme etape du pipeline ChurnML : entrainer un modele
scikit-learn a partir des variables preparees par feature-engineering
et decoupees par train-test-split.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Modeles disponibles pour train()
MODELS = ("random_forest", "logistic_regression")


def train(X, y, model_name="random_forest", seed=42):
    """Entraine un modele de classification et le retourne.

    Args:
        X: DataFrame des variables explicatives (colonnes numeriques uniquement).
        y: Serie cible (churn : 0 ou 1).
        model_name: "random_forest" (defaut) ou "logistic_regression".
        seed: Graine aleatoire (resultat reproductible).

    Returns:
        Le modele scikit-learn entraine (methodes predict / predict_proba).

    Raises:
        ValueError: si model_name est inconnu, si X est vide, si X et y
            n'ont pas la meme taille ou si X contient des colonnes non numeriques.
    """
    if model_name not in MODELS:
        raise ValueError(f"Modele inconnu : {model_name}. Choix : {list(MODELS)}")

    if len(X) == 0:
        raise ValueError("X est vide : entrainement impossible.")

    if len(X) != len(y):
        raise ValueError(
            f"X et y n'ont pas la meme taille ({len(X)} contre {len(y)})."
        )

    non_numeriques = list(X.select_dtypes(exclude="number").columns)
    if non_numeriques:
        raise ValueError(
            f"Colonnes non numeriques a retirer ou encoder : {non_numeriques}"
        )

    if model_name == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=seed)
    else:
        model = LogisticRegression(max_iter=1000, random_state=seed)

    model.fit(X, y)
    return model
