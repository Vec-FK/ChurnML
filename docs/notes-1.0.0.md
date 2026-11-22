## ChurnML v1.0.0

Première version stable du pipeline de prédiction du churn client.

### Fonctionnalités

- data-loading : chargement et validation des données clients
- eda-report : rapport exploratoire (dimensions, manquantes, statistiques, churn)
- feature-engineering : one-hot de contract_type + variables dérivées
- train-test-split : découpage train/test stratifié et reproductible
- model-training : entraînement du modèle de classification
- evaluation-metrics : accuracy, precision, recall, F1
- model-persistence : sauvegarde/rechargement du modèle (joblib)
- predict-cli : CLI de prédiction
- churn-charts : visualisations matplotlib

### Contribution externe

- data-quality : contrôle qualité des données — équipe E4 (SentimentLab), via fork + PR #15

### Installation et tests

```bash id="uy9rvk"
pip install -r requirements.txt
python -m pytest tests/ -v
```