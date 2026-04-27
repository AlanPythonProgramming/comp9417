from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from xrfm.xrfm import xRFM


def model_builder_xrfm(params: Dict[str, Any]) -> xRFM:
    rfm_params = {
        "model": {
            "kernel": "l2",
            "bandwidth": params.get("bandwidth", 3.0),
            "exponent": 1.0,
            "diag": params.get("diag", True),
            "bandwidth_mode": "constant",
        },
        "fit": {
            "reg": params.get("reg", 1e-3),
            "iters": params.get("iters", 1),
            "M_batch_size": params.get("M_batch_size", 8),
            "verbose": False,
            "early_stop_rfm": True,
        },
    }

    return xRFM(
        rfm_params=rfm_params,
        max_leaf_size=params.get("max_leaf_size", 1000),
        n_tree_iters=params.get("n_tree_iters", 0),
        n_trees=1,
        split_method="top_vector_agop_on_subset",
        tuning_metric="mse",
        random_state=42,
        verbose=False,
    )


def fit_fn_xrfm(model, X_tr, y_tr, X_val, y_val):
    X_tr  = np.asarray(X_tr,  dtype=np.float32)
    y_tr  = np.asarray(y_tr,  dtype=np.float32).reshape(-1, 1)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)

    # duplicate target column to satisfy xRFM regression shape bug
    y_tr  = np.concatenate([y_tr,  y_tr],  axis=1)
    y_val = np.concatenate([y_val, y_val], axis=1)

    model.fit(X_tr, y_tr, X_val, y_val)
    return model


def predict_fn_xrfm(model, X):
    X    = np.asarray(X, dtype=np.float32)
    pred = np.asarray(model.predict(X))

    if pred.ndim == 2:
        return pred[:, 0]

    return pred.reshape(-1)


def build_xrfm_preprocessor(X_train, target_col: Optional[str] = None) -> ColumnTransformer:
    cat_cols = [col for col in X_train.columns if X_train[col].dtype == "O"]
    num_cols = [col for col in X_train.columns if col not in cat_cols]

    if target_col is not None and target_col in num_cols:
        num_cols.remove(target_col)
    if target_col is not None and target_col in cat_cols:
        cat_cols.remove(target_col)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def tune_xrfm_regressor(
    X_train,
    y_train,
    X_test,
    y_test,
    target_col: Optional[str] = None,
    n_trials: int = 20,
    n_splits: int = 5,
    param_space: Optional[Dict[str, Dict[str, Any]]] = None,
):
    from hp_script import bayes_tune_model

    if param_space is None:
        param_space = {
            "bandwidth":     {"type": "float",       "low": 0.5, "high": 10.0, "log": True},
            "reg":           {"type": "float",       "low": 1e-5, "high": 1e-2, "log": True},
            "iters":         {"type": "int",         "low": 1,   "high": 5},      # was 10
            "diag":          {"type": "categorical", "choices": [True]},           # was [True, False]
            "M_batch_size":  {"type": "int",         "low": 5,   "high": 50},     # was 100
            "max_leaf_size": {"type": "int",         "low": 100, "high": 500},    # already 500 for regressor
            "n_tree_iters":  {"type": "int",         "low": 0,   "high": 5},      # was 10
        }

    preprocessor = build_xrfm_preprocessor(X_train=X_train, target_col=target_col)
    X_train_processed = np.asarray(preprocessor.fit_transform(X_train), dtype=np.float32)
    X_test_processed  = np.asarray(preprocessor.transform(X_test),      dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_test  = np.asarray(y_test,  dtype=np.float32)

    results = bayes_tune_model(
        X_train=X_train_processed,
        y_train=y_train,
        X_test=X_test_processed,
        y_test=y_test,
        model_builder=model_builder_xrfm,
        fit_fn=fit_fn_xrfm,
        predict_fn=predict_fn_xrfm,
        param_space=param_space,
        n_trials=n_trials,
        n_splits=n_splits,
    )

    return results, preprocessor


def tune_xrfm_classifier(
    X_train,
    y_train,
    X_test,
    y_test,
    target_col: Optional[str] = None,
    n_trials: int = 20,
    n_splits: int = 5,
    param_space: Optional[Dict[str, Dict[str, Any]]] = None,
):
    from hp_script import bayes_tune_model
    from sklearn.metrics import roc_auc_score

    def neg_auc(y_true, y_pred):
        return -roc_auc_score(y_true, y_pred)

    if param_space is None:
        param_space = {
            "bandwidth":     {"type": "float",       "low": 0.5,  "high": 10.0, "log": True},
            "reg":           {"type": "float",       "low": 1e-5, "high": 1e-2, "log": True},
            "iters":         {"type": "int",         "low": 1,    "high": 5},
            "diag":          {"type": "categorical", "choices": [True]},
            "M_batch_size":  {"type": "int",         "low": 5,    "high": 50},
            "max_leaf_size": {"type": "int",         "low": 100,  "high": 500},
            "n_tree_iters":  {"type": "int",         "low": 0,    "high": 5},
        }

    preprocessor = build_xrfm_preprocessor(X_train=X_train, target_col=target_col)
    X_train_processed = np.asarray(preprocessor.fit_transform(X_train), dtype=np.float32)
    X_test_processed  = np.asarray(preprocessor.transform(X_test),      dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_test  = np.asarray(y_test,  dtype=np.float32)

    results = bayes_tune_model(
        X_train=X_train_processed,
        y_train=y_train,
        X_test=X_test_processed,
        y_test=y_test,
        model_builder=model_builder_xrfm,
        fit_fn=fit_fn_xrfm,
        predict_fn=predict_fn_xrfm,
        param_space=param_space,
        metric_fn=neg_auc,
        direction="minimize",
        n_trials=n_trials,
        n_splits=n_splits,
    )

    return results, preprocessor