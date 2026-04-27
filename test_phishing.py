import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# tldr: process pipeline
# 1. loads and preprocesses the phishing dataset
# 2. splits into train/val/test
# 3. tunes xRFM, XGBoost and MLP using shared tuning modules
# 4. results!
# task: CLASSIFICATION - predicting whether a url is phishing or legit

import pandas as pd
import numpy as np
import pickle
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from xgb_tuning import tune_xgb_classifier
from xrfm_tuning import tune_xrfm_classifier
from mlp_tuning import tune_mlp_classifier

# load data from .arff file
print("LOAD DATA")
data, meta = arff.loadarff('datasets/phishing/PhishingData.arff')
df = pd.DataFrame(data)
# bytes -> strings (arff stores strings as bytes)
df = df.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
# columns -> integers
# all features are encoded as -1 (phishing), 0 (sus), or 1 (legit)
df = df.astype(int)

print("Shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print("Target distribution:")
print(df['Result'].value_counts())

# split features
# X = all columns but Result & y = Result
print("\nPREPROCESSING")
X = df.drop('Result', axis=1)
y = df['Result']

# convert target from (-1, 1) to (0, 1) for XGBoost and Keras
y = (y == 1).astype(int)

print("Features shape:", X.shape)
print("Target distribution after conversion:")
print(y.value_counts())

# split into 3 parts:
# - training set (64%): models learn from this
# - validation set (16%): used during tuning
# - test set (20%): only used at the end for final evaluation
# random_state=42 ensures everyone gets the same split
print("\nTRAIN/VAL/TEST SPLIT")

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

print("Training set:", X_train.shape)
print("Validation set:", X_val.shape)
print("Test set:", X_test.shape)

# train all three models using bayesian tuning
# 20 trials each, 5-fold CV on training set, final eval on held-out test set
# tuning metric is neg AUC-ROC (handled inside the classifier tuning functions)
# preprocessing is handled inside each tuning function
print("\nTRAINING MODELS")

os.makedirs('results/phishing', exist_ok=True)

print("\nTuning XGBoost...")
xgb_results, _ = tune_xgb_classifier(
    X_train,
    y_train,
    X_test,
    y_test,
    n_trials=20,
    n_splits=5,
)
xgb_proba = xgb_results["test_predictions"]
xgb_preds = (xgb_proba > 0.5).astype(int)
print("XGBoost best params:", xgb_results["best_params"])
print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
print(f"XGBoost AUC-ROC:  {roc_auc_score(y_test, xgb_proba):.4f}")

with open('results/phishing/xgboost.pkl', 'wb') as f:
    pickle.dump(xgb_results, f)

print("\nTuning xRFM...")
xrfm_results, _ = tune_xrfm_classifier(
    X_train,
    y_train,
    X_test,
    y_test,
    n_trials=20,
    n_splits=5,
)
xrfm_proba = xrfm_results["test_predictions"]
xrfm_preds = (xrfm_proba > 0.5).astype(int)
print("xRFM best params:", xrfm_results["best_params"])
print(f"xRFM Accuracy: {accuracy_score(y_test, xrfm_preds):.4f}")
print(f"xRFM AUC-ROC:  {roc_auc_score(y_test, xrfm_proba):.4f}")

with open('results/phishing/xrfm.pkl', 'wb') as f:
    pickle.dump(xrfm_results, f)

print("\nTuning MLP...")
mlp_results, _ = tune_mlp_classifier(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    n_trials=20,
    n_splits=5,
)
mlp_proba = mlp_results["test_predictions"]
mlp_preds = (mlp_proba > 0.5).astype(int)
print("MLP best params:", mlp_results["best_params"])
print(f"MLP Accuracy: {accuracy_score(y_test, mlp_preds):.4f}")
print(f"MLP AUC-ROC:  {roc_auc_score(y_test, mlp_proba):.4f}")

with open('results/phishing/mlp.pkl', 'wb') as f:
    pickle.dump(mlp_results, f)

# results! accuracy and AUC-ROC, higher is better for both
print("\nFINAL RESULTS: PHISHING DATASET!")
print(f"{'Model':<20} {'Accuracy':<12} {'AUC-ROC':<12} {'Best CV neg-AUC'}")
print("-" * 65)
print(f"{'xRFM':<20} {accuracy_score(y_test, xrfm_preds):<12.4f} {roc_auc_score(y_test, xrfm_proba):<12.4f} {xrfm_results['best_cv_score']:.4f}")
print(f"{'XGBoost':<20} {accuracy_score(y_test, xgb_preds):<12.4f} {roc_auc_score(y_test, xgb_proba):<12.4f} {xgb_results['best_cv_score']:.4f}")
print(f"{'MLP':<20} {accuracy_score(y_test, mlp_preds):<12.4f} {roc_auc_score(y_test, mlp_proba):<12.4f} {mlp_results['best_cv_score']:.4f}")