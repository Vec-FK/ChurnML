# Changelog

Toutes les modifications notables de ChurnML.

Format inspiré de Keep a Changelog ; versionnage sémantique (SemVer).

## [1.0.0] - 2026-11-21

### Added

- data-loading : chargement et validation du CSV clients (@Tiamaz-Md)
- eda-report : résumé exploratoire et rapport markdown (@fikriadam2003)
- feature-engineering : encodage one-hot de contract_type et variables dérivées (@fikriadam2003)
- train-test-split : découpage train/test stratifié et reproductible (@walidblack)
- model-training : entraînement du modèle de classification (@laarw)
- evaluation-metrics : accuracy, precision, recall et F1 (@chajdymountasser)
- model-persistence : sauvegarde et rechargement du modèle avec joblib (@chaimaafadilma-cyber)
- predict-cli : interface en ligne de commande de prédiction (@SalwaN23)
- churn-charts : visualisations matplotlib du churn (@simomn22)
- data-quality : contrôle qualité des données — contribution externe de l'équipe E4 SentimentLab (@s-7a9)

### Fixed

- Résolution des conflits README entre branches de fonctionnalités (GitFlow)