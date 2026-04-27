from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


def model_builder_xgb(params: Dict[str, Any]) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=25,
        **params,
    )


def model_builder_xgb_classifier(params: Dict[str, Any]) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=25,
        **params,
    )


def fit_fn_xgb(model, X_tr, y_tr, X_val, y_val):
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def predict_fn_default(model, X):
    return model.predict(X)


def predict_fn_proba(model, X):
    return model.predict_proba(X)[:, 1]


def build_xgb_preprocessor(X_train, target_col: Optional[str] = None) -> ColumnTransformer:
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


def tune_xgb_regressor(
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
            "n_estimators":     {"type": "int",   "low": 200,  "high": 1000},
            "max_depth":        {"type": "int",   "low": 3,    "high": 8},
            "learning_rate":    {"type": "float", "low": 0.01, "high": 0.2,  "log": True},
            "subsample":        {"type": "float", "low": 0.6,  "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.6,  "high": 1.0},
            "min_child_weight": {"type": "int",   "low": 1,    "high": 10},
            "reg_alpha":        {"type": "float", "low": 1e-8, "high": 2.0,  "log": True},
            "reg_lambda":       {"type": "float", "low": 1e-6, "high": 20.0, "log": True},
        }

    preprocessor = build_xgb_preprocessor(X_train=X_train, target_col=target_col)
    X_train_processed = np.asarray(preprocessor.fit_transform(X_train))
    X_test_processed  = np.asarray(preprocessor.transform(X_test))
    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)

    results = bayes_tune_model(
        X_train=X_train_processed,
        y_train=y_train,
        X_test=X_test_processed,
        y_test=y_test,
        model_builder=model_builder_xgb,
        fit_fn=fit_fn_xgb,
        predict_fn=predict_fn_default,
        param_space=param_space,
        n_trials=n_trials,
        n_splits=n_splits,
    )

    return results, preprocessor


def tune_xgb_classifier(
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
            "n_estimators":     {"type": "int",   "low": 200,  "high": 1000},
            "max_depth":        {"type": "int",   "low": 3,    "high": 8},
            "learning_rate":    {"type": "float", "low": 0.01, "high": 0.2,  "log": True},
            "subsample":        {"type": "float", "low": 0.6,  "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.6,  "high": 1.0},
            "min_child_weight": {"type": "int",   "low": 1,    "high": 10},
            "reg_alpha":        {"type": "float", "low": 1e-8, "high": 2.0,  "log": True},
            "reg_lambda":       {"type": "float", "low": 1e-6, "high": 20.0, "log": True},
        }

    preprocessor = build_xgb_preprocessor(X_train=X_train, target_col=target_col)
    X_train_processed = np.asarray(preprocessor.fit_transform(X_train))
    X_test_processed  = np.asarray(preprocessor.transform(X_test))
    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)

    results = bayes_tune_model(
        X_train=X_train_processed,
        y_train=y_train,
        X_test=X_test_processed,
        y_test=y_test,
        model_builder=model_builder_xgb_classifier,
        fit_fn=fit_fn_xgb,
        predict_fn=predict_fn_proba,
        param_space=param_space,
        metric_fn=neg_auc,
        direction="minimize",
        n_trials=n_trials,
        n_splits=n_splits,
    )

    return results, preprocessor