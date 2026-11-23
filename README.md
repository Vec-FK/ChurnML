# ChurnML - Prédiction du churn client (pandas, scikit-learn, matplotlib)

## Description

ChurnML est un pipeline de prédiction du churn client : chargement et validation des données, analyse exploratoire, ingénierie de variables, découpage train/test, entraînement du modèle, évaluation, sauvegarde du modèle, CLI de prédiction et visualisations.

## Installation

Python 3.10+ recommandé.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Lancer les tests :

```bash
python -m pytest tests/ -v
```

Pipeline complet (dans l'ordre) :

```python
from src.data_loading import load_data
from src.feature_engineering import transform
from src.train_test_split import split_data
from src.model_training import train
from src.evaluation_metrics import evaluate
from src.model_persistence import save_model

df = transform(load_data("data/sample/churn_sample.csv"))
X_train, X_test, y_train, y_test = split_data(df)
model = train(X_train, y_train)
print(evaluate(model, X_test, y_test))
save_model(model, "model.joblib")
```

CLI de prédiction :

```bash
python -m src.predict_cli --model models/model.joblib --input data/sample/churn_sample.csv
```

## Données

Le jeu de test versionné est `data/sample/churn_sample.csv` (12 lignes).

Le jeu complet (200 clients) se génère localement dans `data/raw/churn_full.csv` (jamais versionné, voir `.gitignore`).

Jeu de référence pour un usage réel : IBM Telco Customer Churn (7 043 lignes, licence Apache 2.0) — [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d)

## Fonctionnalités

- data-loading : chargement du CSV clients dans un DataFrame pandas (normalisation + validation des colonnes) [@Tiamaz-Md](https://github.com/Tiamaz-Md)
- eda-report : résumé exploratoire (dimensions, valeurs manquantes, statistiques, distribution du churn) [@fikriadam2003](https://github.com/fikriadam2003)
- feature-engineering : encodage one-hot de contract_type + variables dérivées (total_charges, is_new_customer, support_intensity) [@fikriadam2003](https://github.com/fikriadam2003)
- data-quality : contrôle de qualité des données (check_quality), lance ValueError si churn ou contract_type est absente [@s-7a9](https://github.com/s-7a9)
- train-test-split : découpage train/test reproductible (seed fixe, colonne cible churn) [@walidblack](https://github.com/walidblack)
- model-training : entraînement d'un modèle de classification (random forest ou régression logistique) [@laarw](https://github.com/laarw)
- evaluation-metrics : évaluation du modèle sur le jeu de test (accuracy, precision, recall, F1) [@chajdymountasser](https://github.com/chajdymountasser)
- model-persistence : sauvegarde et rechargement du modèle entraîné avec joblib (save_model, load_model) [@chaimaafadilma-cyber](https://github.com/chaimaafadilma-cyber)
- predict-cli : interface en ligne de commande qui charge un modèle sauvegardé et prédit le churn d'un CSV de clients (probabilité incluse) [@SalwaN23](https://github.com/SalwaN23)
- churn-charts : visualisations du churn (taux par contrat, ancienneté selon le churn) [@simomn22](https://github.com/simomn22)