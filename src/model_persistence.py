"""Sauvegarde du modele (feature model-persistence).

Etape du pipeline ChurnML apres model-training : enregistrer le modele
entraine sur disque avec joblib, puis le recharger pour predire sans
reentrainer.
"""

from pathlib import Path

import joblib


def save_model(model, path):
    """Sauvegarde un modele entraine dans un fichier joblib.

    Args:
        model: Modele scikit-learn deja entraine (methode predict).
        path: Chemin du fichier de sortie (ex : "models/model.joblib").
            Les dossiers parents sont crees s'ils n'existent pas.

    Returns:
        Le chemin (Path) du fichier ecrit.

    Raises:
        ValueError: si model est None ou n'a pas de methode predict.
    """
    if model is None or not hasattr(model, "predict"):
        raise ValueError("Le modele doit etre entraine (methode predict requise).")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path):
    """Charge un modele sauvegarde avec save_model.

    Args:
        path: Chemin du fichier joblib.

    Returns:
        Le modele scikit-learn recharge.

    Raises:
        FileNotFoundError: si le fichier n'existe pas.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Modele introuvable : {path}")

    return joblib.load(path)
