"""
Superconductivity dataset: xRFM vs. XGBoost vs. MLP comparison.

Author: Isabella 
Dataset: UCI Superconductivity (https://archive.ics.uci.edu/dataset/464/superconductivty+data)
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from xgb_tuning import tune_xgb_regressor
from xrfm_tuning import tune_xrfm_regressor
from mlp_tuning import tune_mlp_regressor


# Configuration

DATASET_NAME = "superconductivity"
DATA_PATH = Path("datasets") / DATASET_NAME / "train.csv"
TARGET_COL = "critical_temp"
RESULTS_DIR = Path("results") / DATASET_NAME

N_TRIALS = 2
N_SPLITS = 2
RANDOM_STATE = 42
TEST_SIZE = 0.25

# Flags — lets you run one model at a time without commenting out code
RUN_XGB = True
RUN_XRFM = False
RUN_MLP = True


# Helper: save results so we don't lose work if a later model crashes
def save_results(results_dict, model_name):
    """Save a tuning result dict as pickle + a small text summary."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pkl_path = RESULTS_DIR / f"{model_name}_results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results_dict, f)

    txt_path = RESULTS_DIR / f"{model_name}_summary.txt"
    with open(txt_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {DATASET_NAME}\n")
        f.write(f"Test RMSE: {results_dict['test_rmse']:.4f}\n")
        f.write(f"Test R²:   {results_dict['test_r2']:.4f}\n")
        f.write(f"Best CV score: {results_dict['best_cv_score']:.4f}\n")
        f.write(f"Best params:\n")
        for k, v in results_dict["best_params"].items():
            f.write(f"  {k}: {v}\n")

    print(f"  Saved: {pkl_path.name}, {txt_path.name}")


# Main: load data, run models sequentially, save results as we go
def main():
    print(f"\n{'=' * 60}")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"{'=' * 60}\n")

    # ---- Load and split ----
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=TARGET_COL)
    y = df[TARGET_COL]

    print(f"Rows: {len(df)}, Features: {X.shape[1]}")
    print(f"Categorical: {sum(X.dtypes == 'O')}, "
          f"Numeric: {sum(X.dtypes != 'O')}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    summary_rows = []

    # ---- XGBoost ----
    if RUN_XGB:
        print("-" * 60)
        print("  Tuning XGBoost")
        print("-" * 60)
        t0 = time.time()
        xgb_results, _ = tune_xgb_regressor(
            X_train, y_train, X_test, y_test,
            target_col=TARGET_COL,
            n_trials=N_TRIALS, n_splits=N_SPLITS,
        )
        elapsed = time.time() - t0
        print(f"  Total tuning time: {elapsed:.1f}s")
        print(f"  Test RMSE: {xgb_results['test_rmse']:.4f}")
        print(f"  Test R²:   {xgb_results['test_r2']:.4f}\n")

        save_results(xgb_results, "xgboost")
        summary_rows.append({
            "model": "XGBoost",
            "test_rmse": xgb_results["test_rmse"],
            "test_r2": xgb_results["test_r2"],
            "cv_rmse": xgb_results["best_cv_score"],
            "tuning_time_s": elapsed,
        })

    # ---- xRFM ----
    if RUN_XRFM:
        print("-" * 60)
        print("  Tuning xRFM")
        print("-" * 60)

        xrfm_param_space = {
        "bandwidth":     {"type": "float", "low": 0.5, "high": 10.0, "log": True},
        "reg":           {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "iters":         {"type": "int",   "low": 1, "high": 5},
        "diag":          {"type": "categorical", "choices": [True, False]},
        "M_batch_size":  {"type": "int",   "low": 5, "high": 100},
        "max_leaf_size": {"type": "int",   "low": 100, "high": 500},
        "n_tree_iters":  {"type": "int",   "low": 0, "high": 5},
        }

        t0 = time.time()
        xrfm_results, _ = tune_xrfm_regressor(
            X_train, y_train, X_test, y_test,
            target_col=TARGET_COL,
            n_trials=N_TRIALS, 
            n_splits=N_SPLITS,
            param_space=xrfm_param_space, # to pass our custom param
        )
        elapsed = time.time() - t0
        print(f"  Total tuning time: {elapsed:.1f}s")
        print(f"  Test RMSE: {xrfm_results['test_rmse']:.4f}")
        print(f"  Test R²:   {xrfm_results['test_r2']:.4f}\n")

        save_results(xrfm_results, "xrfm")
        summary_rows.append({
            "model": "xRFM",
            "test_rmse": xrfm_results["test_rmse"],
            "test_r2": xrfm_results["test_r2"],
            "cv_rmse": xrfm_results["best_cv_score"],
            "tuning_time_s": elapsed,
        })

    # ---- MLP ----
    if RUN_MLP:
        print("-" * 60)
        print("  Tuning MLP")
        print("-" * 60)
        t0 = time.time()
        mlp_results, _ = tune_mlp_regressor(
            X_train, y_train, X_test, y_test,
            target_col=TARGET_COL,
            n_trials=N_TRIALS, n_splits=N_SPLITS,
        )
        elapsed = time.time() - t0
        print(f"  Total tuning time: {elapsed:.1f}s")
        print(f"  Test RMSE: {mlp_results['test_rmse']:.4f}")
        print(f"  Test R²:   {mlp_results['test_r2']:.4f}\n")

        save_results(mlp_results, "mlp")
        summary_rows.append({
            "model": "MLP",
            "test_rmse": mlp_results["test_rmse"],
            "test_r2": mlp_results["test_r2"],
            "cv_rmse": mlp_results["best_cv_score"],
            "tuning_time_s": elapsed,
        })

    # ---- Combined summary CSV ----
    summary_df = pd.DataFrame(summary_rows)
    summary_df["dataset"] = DATASET_NAME
    summary_path = RESULTS_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("  Final summary")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print(f"\nSaved to: {summary_path}")


if __name__ == "__main__":
    main()