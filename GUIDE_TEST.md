# Guide de test complet du projet

## 📋 Checklist avant de commencer

- [x] Fichier de données : `city-hall-electricity-use.csv` présent
- [x] Fichiers de preprocessing générés : `data_train.csv`, `data_val.csv`, `data_test.csv`
- [ ] Dépendances installées (vérifier avec `pip install -r requirements.txt`)
- [ ] Modèle optimisé : `lgbm_optimise.pkl` (sera généré à l'étape 4)

---

## 🚀 Ordre d'exécution pour tester TOUT le projet

### Étape 1 : Vérifier les dépendances (si pas déjà fait)

```bash
cd /home/ajinou/Bureau/fille/energy-demand-prediction
pip install -r requirements.txt
```

### Étape 2 : Preprocessing (déjà fait, mais peut être relancé)

```bash
python preprocessing.py
```

**Résultat attendu :**
- ✅ Fichier trouvé : city-hall-electricity-use.csv
- ✅ Données chargées : ~106766 lignes
- ✅ Génération de `data_train.csv`, `data_val.csv`, `data_test.csv`

**Temps estimé :** 1-2 minutes

---

### Étape 3 : Tester les modèles baseline

```bash
python model_baseline.py
```

**Résultat attendu :**
- Régression linéaire: Validation - RMSE: ~34, MAE: ~17
- Régression linéaire: Test - RMSE: ~26.54, MAE: ~13.82
- ARIMA: Test - RMSE: ~298, MAE: ~247
- Génération de `baseline_predictions.png`

**Temps estimé :** 2-3 minutes (ARIMA peut être lent)

---

### Étape 4 : Tester les modèles Machine Learning

```bash
python model_ml.py
```

**Résultat attendu :**
- RandomForest - Test RMSE: ~33.65, MAE: ~16.88
- XGBoost - Test RMSE: ~138.99, MAE: ~69.95
- LightGBM - Test RMSE: ~34.90, MAE: ~19.35
- Génération de `ml_predictions.png`

**Temps estimé :** 5-10 minutes (selon votre machine)

---

### Étape 5 : Tester le modèle LSTM (optionnel, plus long)

```bash
python model_lstm.py
```

**Résultat attendu :**
- LSTM - Test RMSE: ~40.73, MAE: ~23.34
- Génération de `lstm_predictions.png`

**Temps estimé :** 10-20 minutes (peut varier selon GPU/CPU)

---

### Étape 6 : Optimisation du modèle LightGBM (IMPORTANT pour l'app web)

```bash
python optimisation.py
```

**Résultat attendu :**
- Affichage des meilleurs hyperparamètres
- LightGBM optimisé - Test RMSE: ~31.61, MAE: ~16.68
- Génération de `lgbm_optimise.pkl` ⚠️ **NÉCESSAIRE pour l'app web**

**Temps estimé :** 15-30 minutes (GridSearchCV avec cross-validation)

---

### Étape 7 : Comparaison des modèles

```bash
python compare_models.py
```

**Résultat attendu :**
- Tableau récapitulatif des scores
- Génération de `compare_models.png`

**Temps estimé :** 2-3 minutes

---

### Étape 8 : Évaluation et visualisations détaillées

```bash
python eval_visualisation.py
```

**Résultat attendu :**
- Génération de plusieurs visualisations :
  - `zoom_predictions.png`
  - `error_distribution.png`
  - `feature_importance_rf.png`

**Temps estimé :** 2-3 minutes

---

### Étape 9 : Tester l'application web Streamlit

```bash
streamlit run app.py
```

**Résultat attendu :**
- L'application se lance dans votre navigateur (http://localhost:8501)
- Vous pouvez naviguer entre les différentes sections :
  - Accueil / Présentation
  - Visualisation des données
  - Comparaison des modèles
  - Importance des variables
  - Prédiction personnalisée (nécessite `lgbm_optimise.pkl`)
  - Téléchargement des résultats
  - Bonus (Prophet, PCA)

**Pour arrêter :** Appuyez sur `Ctrl+C` dans le terminal

---

## ⚡ Version rapide (sans LSTM)

Si vous voulez tester rapidement sans attendre le LSTM :

```bash
# 1. Preprocessing (déjà fait)
python preprocessing.py

# 2. Baseline
python model_baseline.py

# 3. ML models
python model_ml.py

# 4. Optimisation (important pour l'app)
python optimisation.py

# 5. Comparaison
python compare_models.py

# 6. Visualisations
python eval_visualisation.py

# 7. App web
streamlit run app.py
```

**Temps total estimé :** 30-45 minutes

---

## 🐛 En cas de problème

### Erreur "Module not found"
```bash
pip install -r requirements.txt
```

### Erreur "File not found"
- Vérifiez que `city-hall-electricity-use.csv` est à la racine du projet
- Relancez `preprocessing.py` si nécessaire

### Erreur avec l'app web "lgbm_optimise.pkl not found"
- Exécutez d'abord `python optimisation.py`

### LSTM trop lent
- Vous pouvez sauter cette étape, elle n'est pas nécessaire pour le reste

---

## ✅ Checklist finale

Après avoir tout testé, vous devriez avoir :

- [ ] `data_train.csv`, `data_val.csv`, `data_test.csv`
- [ ] `baseline_predictions.png`
- [ ] `ml_predictions.png`
- [ ] `lstm_predictions.png` (optionnel)
- [ ] `compare_models.png`
- [ ] `zoom_predictions.png`
- [ ] `error_distribution.png`
- [ ] `feature_importance_rf.png`
- [ ] `lgbm_optimise.pkl` ⚠️ **Important pour l'app web**

---

## 🎯 Test rapide de validation

Pour vérifier que tout fonctionne rapidement :

```bash
# Test 1 : Preprocessing
python preprocessing.py

# Test 2 : Un modèle simple
python model_baseline.py

# Test 3 : L'app web (si lgbm_optimise.pkl existe)
streamlit run app.py
```

Si ces 3 étapes fonctionnent, le projet est opérationnel ! 🎉

