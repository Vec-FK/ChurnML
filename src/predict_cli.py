"""CLI de prediction (feature predict-cli).

Derniere etape du pipeline ChurnML : charger un modele sauvegarde,
l'appliquer a un CSV de clients et afficher la prediction de churn.

Exemple :
    python -m src.predict_cli --model models/model.joblib --input data/sample/churn_sample.csv
"""

import argparse

from src.data_loading import load_data
from src.feature_engineering import transform
from src.model_persistence import load_model


def build_parser():
    """Construit le parseur d'arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Predit le churn des clients a partir d'un modele sauvegarde."
    )
    parser.add_argument("--model", required=True, help="Chemin du modele joblib")
    parser.add_argument("--input", required=True, help="CSV des clients")
    parser.add_argument("--output", help="CSV de sortie (optionnel)")
    return parser


def predict(model, df):
    """Retourne un DataFrame customer_id, churn_prediction, churn_probability."""
    features = transform(df)
    X = features[list(model.feature_names_in_)]

    result = df[["customer_id"]].copy()
    result["churn_prediction"] = model.predict(X)
    if hasattr(model, "predict_proba"):
        result["churn_probability"] = model.predict_proba(X)[:, 1].round(3)
    return result


def main(argv=None):
    """Point d'entree de la CLI."""
    args = build_parser().parse_args(argv)

    model = load_model(args.model)
    df = load_data(args.input)
    result = predict(model, df)

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Predictions ecrites dans {args.output}")
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()