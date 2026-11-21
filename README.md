# ChurnML - Prediction du churn client (pandas, scikit-learn, matplotlib)

## Fonctionnalites

- data-loading : chargement du CSV clients dans un DataFrame pandas (normalisation + validation des colonnes) [@Tiamaz-Md](https://github.com/Tiamaz-Md)
- eda-report : resume exploratoire (dimensions, valeurs manquantes, statistiques, distribution du churn) [@fikriadam2003](https://github.com/fikriadam2003)
- feature-engineering : encodage one-hot de contract_type + variables derivees (total_charges, is_new_customer, support_intensity) [@fikriadam2003](https://github.com/fikriadam2003)
- data-quality : controle de qualite des donnees (check_quality), lance ValueError si churn ou contract_type est absente [@s-7a9](https://github.com/s-7a9)
- train-test-split : decoupage train/test reproductible (seed fixe, colonne cible churn) [@walidblack](https://github.com/walidblack)
- model-training : entrainement d'un modele de classification (random forest ou regression logistique) [@laarw](https://github.com/laarw)
- evaluation-metrics : evaluation du modele sur le jeu de test (accuracy, precision, recall, F1) [@chajdymountasser](https://github.com/chajdymountasser)
- model-persistence : sauvegarde et rechargement du modele entraine avec joblib (save_model, load_model) [@chaimaafadilma-cyber](https://github.com/chaimaafadilma-cyber)
- churn-charts : visualisations du churn (taux par contrat, anciennete selon le churn) [@simomn22](https://github.com/simomn22)
