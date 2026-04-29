import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# tldr: process pipeline
# 1. loads and preprocesses the bike sharing dataset
# 2. splits into train/val/test
# 3. tunes xRFM, XGBoost and MLP using shared tuning modules
# 4. results!
# task: REGRESSION - predicting cnt (total bike rentals)

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from xgb_tuning import tune_xgb_regressor
from xrfm_tuning import tune_xrfm_regressor
from mlp_tuning import tune_mlp_regressor

# hour.csv has 17,379 rows of hourly bike rental data
# day.csv only has 731 rows so we use hour.csv
print("LOAD DATA")
df = pd.read_csv('datasets/bike_sharing/bikeSharingData.csv')

print("Shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())

# drop columns that shouldnt be used as features:
# - instant: just a row index
# - dteday: date, already captured by season/month/year
# - casual: adds up to cnt (target) so keeping it is cheating
# - registered: same problem, adds up to cnt
print("\nPREPROCESSING")
df = df.drop(columns=['instant', 'dteday', 'casual', 'registered'])

print("Shape after dropping columns:", df.shape)

# X = all columns but cnt & y = cnt
target = 'cnt'
X = df.drop(columns=target)
y = df[target]

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
# preprocessing is handled inside each tuning function
print("\nTRAINING MODELS")

os.makedirs('results/bike_sharing', exist_ok=True)

print("\nTuning XGBoost...")
xgb_results, _ = tune_xgb_regressor(
    X_train,
    y_train,
    X_test,
    y_test,
    target_col=target,
    n_trials=20,
    n_splits=5,
)
print("XGBoost best params:", xgb_results["best_params"])
print(f"XGBoost test RMSE: {xgb_results['test_rmse']:.4f}")
print(f"XGBoost test R2:   {xgb_results['test_r2']:.4f}")

with open('results/bike_sharing/xgboost.pkl', 'wb') as f:
    pickle.dump(xgb_results, f)

print("\nTuning xRFM...")
xrfm_results, _ = tune_xrfm_regressor(
    X_train,
    y_train,
    X_test,
    y_test,
    target_col=target,
    n_trials=20,
    n_splits=5,
)
print("xRFM best params:", xrfm_results["best_params"])
print(f"xRFM test RMSE: {xrfm_results['test_rmse']:.4f}")
print(f"xRFM test R2:   {xrfm_results['test_r2']:.4f}")

with open('results/bike_sharing/xrfm.pkl', 'wb') as f:
    pickle.dump(xrfm_results, f)

print("\nTuning MLP...")
mlp_results, _ = tune_mlp_regressor(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    target_col=target,
    n_trials=20,
    n_splits=5,
)
print("MLP best params:", mlp_results["best_params"])
print(f"MLP test RMSE: {mlp_results['test_rmse']:.4f}")
print(f"MLP test R2:   {mlp_results['test_r2']:.4f}")

with open('results/bike_sharing/mlp.pkl', 'wb') as f:
    pickle.dump(mlp_results, f)

# results! RMSE = lower is better, R2 = higher is better (1.0 is perfect)
print("\nFINAL RESULTS: BIKE SHARING DATASET!")
print(f"{'Model':<20} {'Test RMSE':<12} {'Test R2':<10} {'Best CV RMSE'}")
print("-" * 60)
print(f"{'xRFM':<20} {xrfm_results['test_rmse']:<12.4f} {xrfm_results['test_r2']:<10.4f} {xrfm_results['best_cv_score']:.4f}")
print(f"{'XGBoost':<20} {xgb_results['test_rmse']:<12.4f} {xgb_results['test_r2']:<10.4f} {xgb_results['best_cv_score']:.4f}")
print(f"{'MLP':<20} {mlp_results['test_rmse']:<12.4f} {mlp_results['test_r2']:<10.4f} {mlp_results['best_cv_score']:.4f}")