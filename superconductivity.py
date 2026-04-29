import os
import pandas as pd 
import numpy as np 
import pickle
import time
from pathlib import Path

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from xgb_tuning import tune_xgb_regressor
from xrfm_tuning import tune_xrfm_regressor
from mlp_tuning import tune_mlp_regressor

df = pd.read_csv('datasets/superconductivity/train.csv')
target = "critical_temp"
X = df.drop(columns=target)
y = df[target]

print(f"Rows: {len(df)}, Features: {X.shape[1]}")
print(f"Target stats: {y.describe()}")

X_temp, X_test, y_temp, y_test, = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

os.makedirs('results/superconductivity', exist_ok=True)

print("\nTuning XGBoost")
xgb_results, _ = tune_xgb_regressor(
            X_train, y_train, X_test, y_test,
            target_col=target,
            n_trials=20, n_splits=5, )
print("XGBoost best params:", xgb_results["best_params"])
print(f"XGBoost test RMSE: {xgb_results['test_rmse']:.4f}")
print(f"XGBoost test R2:   {xgb_results['test_r2']:.4f}")

with open("results/superconductivity/xgboost.pkl", 'wb') as f:
    pickle.dump(xgb_results, f)


xrfm_param_space = {
    "bandwidth":     {"type": "float", "low": 1.0, "high": 5.0, "log": True},
    "reg":           {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
    "iters":         {"type": "int",   "low": 2, "high": 4},     # narrower
    "diag":          {"type": "categorical", "choices": [True]},  # fix to True
    "M_batch_size":  {"type": "fixed", "value": 50},              # fix
    "max_leaf_size": {"type": "int",   "low": 100, "high": 300},
    "n_tree_iters":  {"type": "int",   "low": 0, "high": 2},      # narrower
}

print("\nTuning xRFM")
xrfm_results, _ = tune_xrfm_regressor(
            X_train, y_train, X_test, y_test,
            target_col=target,
            n_trials=1, n_splits=3, param_space=xrfm_param_space,)
print("xRFM best params:", xrfm_results["best_params"])
print(f"xRFM test RMSE: {xrfm_results['test_rmse']:.4f}")
print(f"xRFM test R2:   {xrfm_results['test_r2']:.4f}")

with open("results/superconductivity/xrfm.pkl", 'wb') as f:
    pickle.dump(xrfm_results, f)


print("\nTuning MLP")
mlp_results, _ = tune_mlp_regressor(
            X_train, y_train, X_test, y_test,
            target_col=target,
            n_trials=20, n_splits=5, )
print("MLP best params:", mlp_results["best_params"])
print(f"MLP test RMSE: {mlp_results['test_rmse']:.4f}")
print(f"MLP test R2:   {mlp_results['test_r2']:.4f}")

with open("results/superconductivity/mlp.pkl", 'wb') as f:
    pickle.dump(mlp_results, f)

# Results
print("\nResults for Superconductivity Dataset")
print(f"{'Model':<20}{'Test RMSE':<12}{'Test R2':<10}{'Best CV RMSE'}")
print(f"{'xRFM':<20}{xrfm_results['test_rmse']:<12.4f}{xrfm_results['test_r2']:<10.4f}{xrfm_results['best_cv_score']:.4f}")
print(f"{'XGBoost':<20}{xgb_results['test_rmse']:<12.4f}{xgb_results['test_r2']:<10.4f}{xgb_results['best_cv_score']:.4f}")
print(f"{'MLP':<20}{mlp_results['test_rmse']:<12.4f}{mlp_results['test_r2']:<10.4f}{mlp_results['best_cv_score']:.4f}")