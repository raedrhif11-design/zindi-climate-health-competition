# Climate Risk Health Prediction Challenge - Zindi

Solution de Machine Learning pour prédire l'impact des variations climatiques sur la santé publique dans le cadre de la compétition Zindi.

## Architecture & Méthodologie
- **Modèle :** LightGBM Classifier (Gradient Boosted Trees)
- **Validation croisée :** Stratified K-Fold (5 plis)
- **Cible :** Classification binaire (`is_climate_sensitive`)
- **Évaluation :**
  - **ROC-AUC (CV locale) :** ~0.8118
  - **F1-Score (CV locale) :** ~0.8107 (avec recherche de seuil optimal à 0.40)
- **Outputs générés :**
  - `TargetF1` : Prédictions binaires optimisées
  - `TargetRAUC` : Probabilités continues

## Utilisation
```bash
pip install pandas numpy scikit-learn lightgbm
python zindi.py# zindi-climate-health-competition
