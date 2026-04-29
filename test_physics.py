import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# tldr: process pipeline
# 1. loads the lattice physics dataset from UCI
# 2. splits into train/val/test
# 3. tunes xRFM, XGBoost and MLP using shared tuning modules
# 4. results!
# task: REGRESSION - predicting k-inf (infinite multiplication factor)
# dataset: 24,000 samples, 39 features (fuel pin enrichments), no missing values
# all features are real-valued so no categorical encoding needed

import pandas as pd
import numpy as np
import pickle
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from xgb_tuning import tune_xgb_regressor
from xrfm_tuning import tune_xrfm_regressor
from mlp_tuning import tune_mlp_regressor

# fetch dataset directly from UCI — no manual download needed
print("LOAD DATA")
df = pd.read_csv('datasets/lattice_physics/raw.csv', sep=r'\s+', header=None)

# 39 fuel pin enrichment features + 2 targets (k-inf, PPPF) = 41 columns
# last two columns are the targets
X = df.iloc[:, :39]
y = df.iloc[:, 39]  # k-inf is column 39

print("Features shape:", X.shape)
print("Target stats:")
print(y.describe())

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
# preprocessing handled inside each tuning function
# all features are numerical so no categorical encoding happens
print("\nTRAINING MODELS")

os.makedirs('results/lattice_physics', exist_ok=True)

print("\nTuning XGBoost...")
xgb_results, _ = tune_xgb_regressor(
    X_train,
    y_train,
    X_test,
    y_test,
    n_trials=20,
    n_splits=5,
)
print("XGBoost best params:", xgb_results["best_params"])
print(f"XGBoost test RMSE: {xgb_results['test_rmse']:.6f}")
print(f"XGBoost test R2:   {xgb_results['test_r2']:.6f}")

with open('results/lattice_physics/xgboost.pkl', 'wb') as f:
    pickle.dump(xgb_results, f)

print("\nTuning xRFM...")
xrfm_results, _ = tune_xrfm_regressor(
    X_train,
    y_train,
    X_test,
    y_test,
    n_trials=20,
    n_splits=5,
)
print("xRFM best params:", xrfm_results["best_params"])
print(f"xRFM test RMSE: {xrfm_results['test_rmse']:.6f}")
print(f"xRFM test R2:   {xrfm_results['test_r2']:.6f}")

with open('results/lattice_physics/xrfm.pkl', 'wb') as f:
    pickle.dump(xrfm_results, f)

print("\nTuning MLP...")
mlp_results, _ = tune_mlp_regressor(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    n_trials=20,
    n_splits=5,
)
print("MLP best params:", mlp_results["best_params"])
print(f"MLP test RMSE: {mlp_results['test_rmse']:.6f}")
print(f"MLP test R2:   {mlp_results['test_r2']:.6f}")

with open('results/lattice_physics/mlp.pkl', 'wb') as f:
    pickle.dump(mlp_results, f)

# results! RMSE = lower is better, R2 = higher is better (1.0 is perfect)
print("\nFINAL RESULTS: LATTICE PHYSICS DATASET!")

for name, r in [('xgboost', xgb_results), ('xrfm', xrfm_results), ('mlp', mlp_results)]:
    print(f"\n=== {name.upper()} ===")
    print("best_params:", r['best_params'])
    print("best_cv_score:", r['best_cv_score'])
    print("test_rmse:", r['test_rmse'])
    print("test_r2:", r['test_r2'])