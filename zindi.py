import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# 1. Chargement
print("Chargement des données...")
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
sub = pd.read_csv('SampleSubmission.csv')

target_col = 'is_climate_sensitive'
id_col = 'ID'
features = [c for c in train.columns if c in test.columns and c != id_col]

# 2. Prétraitement & Feature Engineering
df_all = pd.concat([train[features], test[features]], axis=0).reset_index(drop=True)

# Extraction temporelle
if 'deathdate' in df_all.columns:
    df_all['deathdate'] = pd.to_datetime(df_all['deathdate'])
    df_all['year'] = df_all['deathdate'].dt.year
    df_all['month'] = df_all['deathdate'].dt.month
    df_all['day'] = df_all['deathdate'].dt.day
    df_all['dayofweek'] = df_all['deathdate'].dt.dayofweek
    df_all.drop(columns=['deathdate'], inplace=True)

# Features climatiques simples (amplitude thermique)
if 'max_temperature' in df_all.columns and 'min_temperature' in df_all.columns:
    df_all['temp_range'] = df_all['max_temperature'] - df_all['min_temperature']

# Encodage des catégorielles
cat_cols = df_all.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    df_all[col] = df_all[col].astype('category').cat.codes

X = df_all.iloc[:len(train)].copy()
y = train[target_col].astype(int)
X_test = df_all.iloc[len(train):].copy()

# 3. Stratified K-Fold (Classification binaire)
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_probs = np.zeros(len(train))
test_probs = np.zeros(len(test))

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'random_state': 42,
    'n_estimators': 1500,
    'verbose': -1
}

print("Entraînement en classification binaire...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_f, y_train_f = X.iloc[train_idx], y.iloc[train_idx]
    X_val_f, y_val_f = X.iloc[val_idx], y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train_f, y_train_f,
        eval_set=[(X_val_f, y_val_f)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    val_pred = model.predict_proba(X_val_f)[:, 1]
    oof_probs[val_idx] = val_pred
    test_probs += model.predict_proba(X_test)[:, 1] / n_splits
    
    fold_auc = roc_auc_score(y_val_f, val_pred)
    print(f"Fold {fold+1} terminé - ROC-AUC : {fold_auc:.4f}")

global_auc = roc_auc_score(y, oof_probs)
print(f"\n=> CV Global ROC-AUC : {global_auc:.4f}")

# Recherche du seuil optimal pour maximiser le F1-Score
best_thresh = 0.5
best_f1 = 0
for t in np.arange(0.1, 0.9, 0.02):
    score = f1_score(y, (oof_probs > t).astype(int))
    if score > best_f1:
        best_f1 = score
        best_thresh = t

print(f"=> Seuil optimal trouvé : {best_thresh:.2f} avec F1-Score : {best_f1:.4f}")

# 4. Remplissage exact des colonnes attendues
sub['TargetF1'] = (test_probs > best_thresh).astype(int)
sub['TargetRAUC'] = test_probs

sub.to_csv('submission_lgbm.csv', index=False)
print("\nFichier 'submission_lgbm.csv' généré avec succès avec TargetF1 et TargetRAUC !")