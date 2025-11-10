# Prédiction de la demande d'énergie renouvelable

## 1. Problématique et Contexte
Avec l'essor des énergies renouvelables, prévoir la demande d'énergie est crucial pour optimiser la production et la distribution. Ce projet vise à prédire la consommation d'électricité à partir de données réelles, en utilisant des techniques de machine learning et deep learning.

## 2. Données
- **Source** : [City Hall Electricity Usage – Boston](https://data.boston.gov/dataset/city-hall-electricity-usage)
- **Description** :
  - Consommation électrique mesurée toutes les 15 minutes à la mairie de Boston (2016-2020)
  - Variable cible : `Total_Demand_KW`
  - Variables explicatives créées : heure, jour de la semaine, mois, saison, week-end, lags temporels

### 📥 Comment obtenir les données

**IMPORTANT** : Les fichiers de données ne sont pas inclus dans ce dépôt. Vous devez les télécharger vous-même.

1. **Télécharger le fichier source** :
   - Allez sur [City Hall Electricity Usage – Boston](https://data.boston.gov/dataset/city-hall-electricity-usage)
   - Téléchargez le fichier CSV (généralement nommé `city-hall-electricity-use.csv` ou similaire)
   - Placez-le à la racine du projet avec le nom exact : `city-hall-electricity-use.csv`

2. **Format attendu** :
   - Le fichier doit contenir au minimum les colonnes :
     - `DateTime_Measured` : Date et heure au format datetime
     - `Total_Demand_KW` : Consommation électrique en kilowatts

3. **Alternative** : Si vous avez vos propres données de consommation électrique :
   - Assurez-vous qu'elles respectent le format ci-dessus
   - Renommez votre fichier en `city-hall-electricity-use.csv`
   - Placez-le à la racine du projet

## 3. Pipeline de traitement
1. **Collecte et exploration des données**
2. **Prétraitement** :
   - Nettoyage (valeurs nulles, doublons, interpolation)
   - Feature engineering (variables temporelles, lags)
   - Normalisation/standardisation
   - Split train/validation/test (70/15/15)
3. **Modélisation** :
   - Baseline : Régression linéaire, ARIMA
   - Machine Learning : RandomForest, XGBoost, LightGBM
   - Deep Learning : LSTM
4. **Évaluation** :
   - Métriques : RMSE, MAE, MAPE
   - Visualisations : courbes de prédiction, distribution des erreurs, importance des variables
5. **Optimisation** :
   - Tuning d'hyperparamètres (GridSearchCV sur LightGBM)
   - Amélioration des features
6. **Interface web** :
   - Application Streamlit complète
   - Visualisations interactives
   - Prédiction personnalisée
   - Fonctionnalités bonus (Prophet, PCA)

## 4. Résultats
| Modèle                | RMSE   | MAE   | MAPE (%) |
|-----------------------|--------|-------|----------|
| Régression linéaire   | 26.54  | 13.82 | 1.29     |
| RandomForest          | 33.65  | 16.88 | 1.71     |
| XGBoost               | 138.99 | 69.95 | 8.49     |
| LightGBM (défaut)     | 34.90  | 19.35 | 2.08     |
| LightGBM (optimisé)   | 31.61  | 16.68 | -        |
| LSTM                  | 40.73  | 23.34 | -        |

- **La régression linéaire reste la plus performante** sur ce jeu de données.
- **RandomForest et LightGBM** donnent de bons résultats.
- **XGBoost** sous-performe (à optimiser).
- **LSTM** fonctionne mais n'apporte pas de gain ici.

## 5. Interface Web Streamlit

### 🚀 Lancement de l'application
```bash
streamlit run app.py
```

### 📱 Fonctionnalités disponibles
1. **Accueil/Présentation** : Vue d'ensemble du projet
2. **Visualisation des données** : Exploration interactive des données d'entraînement
3. **Comparaison des modèles** : Tableau comparatif des performances
4. **Importance des variables** : Analyse des features les plus importantes
5. **Prédiction personnalisée** : Interface pour faire des prédictions avec le modèle LightGBM optimisé
6. **Téléchargement des résultats** : Export des visualisations et résultats
7. **Bonus - Analyse avancée** :
   - **Prophet** : Prévision de séries temporelles avec saisonnalités
   - **PCA** : Réduction de dimension et visualisation des patterns cachés

## 6. Visualisations clés
- `compare_models.png` : Comparaison des prédictions de chaque modèle
- `zoom_predictions.png` : Zoom sur les 500 premières prédictions
- `error_distribution.png` : Distribution des erreurs
- `feature_importance_rf.png` : Importance des variables (RandomForest)

## 7. Fonctionnalités Bonus

### Prophet - Prévision avancée
- Modèle développé par Facebook pour les séries temporelles
- Gestion automatique des saisonnalités et jours fériés
- Interface pour choisir l'horizon de prévision
- Visualisation des composantes (tendance, saisonnalité)

### PCA - Réduction de dimension
- Analyse en Composantes Principales
- Visualisation des deux premières composantes
- Analyse de la variance expliquée
- Détection de patterns cachés dans les données

## 8. Défis rencontrés et solutions
- **Valeurs nulles et doublons** : Interpolation linéaire et agrégation par timestamp
- **Données bruitées** : Ajout de lags et de variables temporelles pour capter les patterns
- **Modèles deep learning** : Moins performants que les modèles classiques sur ce jeu de données
- **Interface web** : Intégration fluide de toutes les fonctionnalités dans Streamlit

## 9. Perspectives et améliorations
- Ajouter des données météo (température, vent, ensoleillement) pour enrichir les variables explicatives
- Tester des modèles avancés (Transformers pour séries temporelles)
- Déploiement cloud de l'application web
- Ajout de fonctionnalités de monitoring en temps réel
- Intégration de nouveaux datasets pour validation croisée

## 10. Structure du projet
- `devbook.md` : Suivi détaillé du projet étape par étape
- `preprocessing.py` : Pipeline de préparation des données
- `model_baseline.py` : Baseline (régression linéaire, ARIMA)
- `model_ml.py` : Modèles ML (RandomForest, XGBoost, LightGBM)
- `model_lstm.py` : Modèle LSTM
- `compare_models.py` : Comparaison des modèles
- `eval_visualisation.py` : Évaluation détaillée et visualisations
- `optimisation.py` : Tuning d'hyperparamètres
- `app.py` : Application web Streamlit
- `lgbm_optimise.pkl` : Modèle LightGBM optimisé sauvegardé

## 11. Installation et utilisation

### Prérequis
```bash
pip install -r requirements.txt
```

Ou manuellement :
```bash
pip install pandas numpy scikit-learn matplotlib seaborn lightgbm xgboost tensorflow streamlit prophet joblib statsmodels
```

### 📋 Ordre d'exécution des scripts

**IMPORTANT** : Suivez cet ordre pour exécuter le projet correctement :

1. **Télécharger les données** (voir section 2 ci-dessus)
   - Placez `city-hall-electricity-use.csv` à la racine du projet

2. **Prétraitement des données** :
   ```bash
   python preprocessing.py
   ```
   - Génère : `data_train.csv`, `data_val.csv`, `data_test.csv`

3. **Entraînement des modèles** (dans n'importe quel ordre) :
   ```bash
   python model_baseline.py      # Baseline (régression linéaire, ARIMA)
   python model_ml.py            # Modèles ML (RandomForest, XGBoost, LightGBM)
   python model_lstm.py          # Modèle LSTM (peut prendre du temps)
   ```

4. **Optimisation du modèle** :
   ```bash
   python optimisation.py
   ```
   - Génère : `lgbm_optimise.pkl` (nécessaire pour l'application web)

5. **Comparaison et visualisation** :
   ```bash
   python compare_models.py
   python eval_visualisation.py
   ```

6. **Lancer l'application web** :
   ```bash
   streamlit run app.py
   ```
   - Ouvrez votre navigateur sur l'URL indiquée (généralement http://localhost:8501)

### ⚠️ Notes importantes
- Les fichiers de données (`data_train.csv`, `data_val.csv`, `data_test.csv`) et le modèle (`lgbm_optimise.pkl`) sont générés automatiquement lors de l'exécution des scripts
- Si vous modifiez les données source, relancez `preprocessing.py` pour régénérer les datasets
- L'application web nécessite que `lgbm_optimise.pkl` existe (généré par `optimisation.py`)

## 12. Auteur
**Abderrahman AJINOU** – Université Paris Cité  
N° Étudiant : 22116322 – abderrahman.ajinou@etu.u-paris.fr

### Objectifs académiques
- Maîtrise des concepts d'informatique (POO, interfaces graphiques)
- Efficacité des algorithmes et complexité
- Préparation aux masters IA et Cybersécurité
- Ambition : CAIO ou CTO 