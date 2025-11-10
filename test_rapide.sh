#!/bin/bash

# Script de test rapide du projet
# Usage: bash test_rapide.sh

echo "🚀 Démarrage des tests du projet energy-demand-prediction"
echo "=================================================="
echo ""

# Vérification des dépendances
echo "📦 Étape 1/7 : Vérification des dépendances..."
python -c "import pandas, numpy, sklearn, matplotlib, lightgbm, xgboost, tensorflow, streamlit, joblib, statsmodels" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Dépendances principales OK"
else
    echo "⚠️  Certaines dépendances manquent. Installation..."
    pip install -r requirements.txt
fi
echo ""

# Test preprocessing
echo "📊 Étape 2/7 : Test du preprocessing..."
if [ -f "city-hall-electricity-use.csv" ]; then
    echo "✅ Fichier de données trouvé"
    if [ ! -f "data_train.csv" ]; then
        echo "   → Exécution du preprocessing..."
        python preprocessing.py
    else
        echo "✅ Fichiers de preprocessing déjà générés"
    fi
else
    echo "❌ ERREUR : city-hall-electricity-use.csv introuvable !"
    exit 1
fi
echo ""

# Test baseline
echo "🔬 Étape 3/7 : Test du modèle baseline..."
python model_baseline.py 2>&1 | tail -5
if [ -f "baseline_predictions.png" ]; then
    echo "✅ Baseline terminé"
else
    echo "⚠️  baseline_predictions.png non généré"
fi
echo ""

# Test ML
echo "🤖 Étape 4/7 : Test des modèles ML..."
python model_ml.py 2>&1 | tail -5
if [ -f "ml_predictions.png" ]; then
    echo "✅ Modèles ML terminés"
else
    echo "⚠️  ml_predictions.png non généré"
fi
echo ""

# Test optimisation
echo "⚙️  Étape 5/7 : Optimisation LightGBM (peut prendre 15-30 min)..."
if [ ! -f "lgbm_optimise.pkl" ]; then
    echo "   → Démarrage de l'optimisation..."
    python optimisation.py 2>&1 | tail -10
    if [ -f "lgbm_optimise.pkl" ]; then
        echo "✅ Modèle optimisé généré"
    else
        echo "⚠️  lgbm_optimise.pkl non généré"
    fi
else
    echo "✅ Modèle optimisé déjà présent"
fi
echo ""

# Test comparaison
echo "📈 Étape 6/7 : Comparaison des modèles..."
python compare_models.py 2>&1 | tail -10
if [ -f "compare_models.png" ]; then
    echo "✅ Comparaison terminée"
else
    echo "⚠️  compare_models.png non généré"
fi
echo ""

# Résumé
echo "📋 Étape 7/7 : Résumé des fichiers générés..."
echo ""
echo "Fichiers de données :"
ls -lh data_*.csv 2>/dev/null | awk '{print "  ✅", $9, "(" $5 ")"}'
echo ""
echo "Modèles :"
ls -lh *.pkl 2>/dev/null | awk '{print "  ✅", $9, "(" $5 ")"}' || echo "  ⚠️  Aucun modèle .pkl trouvé"
echo ""
echo "Visualisations :"
ls -lh *.png 2>/dev/null | awk '{print "  ✅", $9}' | head -10
echo ""
echo "🎉 Tests terminés !"
echo ""
echo "Pour lancer l'application web :"
echo "  streamlit run app.py"
echo ""

