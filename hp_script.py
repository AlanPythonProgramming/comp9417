import numpy as np
import optuna
from copy import deepcopy


def make_kfold_splits(n_samples, n_splits=5, shuffle=True, random_state=42):
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    splits = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        splits.append((train_idx, val_idx))
        current = stop

    return splits


def suggest_from_space(trial, param_name, spec):
    """
    spec examples:
    {"type": "float", "low": 1e-4, "high": 1e-1, "log": True}
    {"type": "int", "low": 32, "high": 256, "log": False}
    {"type": "categorical", "choices": ["relu", "tanh"]}
    {"type": "fixed", "value": 64}
    """
    ptype = spec["type"]

    if ptype == "float":
        return trial.suggest_float(
            param_name,
            spec["low"],
            spec["high"],
            log=spec.get("log", False)
        )

    if ptype == "int":
        return trial.suggest_int(
            param_name,
            spec["low"],
            spec["high"],
            log=spec.get("log", False)
        )

    if ptype == "categorical":
        return trial.suggest_categorical(param_name, spec["choices"])

    if ptype == "fixed":
        return spec["value"]

    raise ValueError(f"Unsupported parameter type for {param_name}: {ptype}")


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def r2_score_manual(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - ss_res / ss_tot


def bayes_tune_model(
    X_train,
    y_train,
    X_test,
    y_test,
    model_builder,
    fit_fn,
    predict_fn,
    param_space,
    preprocess_fn=None,
    metric_fn=rmse,
    direction="minimize",
    n_trials=30,
    n_splits=5,
    shuffle=True,
    random_state=42,
    study_name=None,
    return_study=False
):
    """
    Generic Bayesian hyperparameter tuner.

    Parameters
    ----------
    model_builder : callable
        model_builder(params) -> model instance

    fit_fn : callable
        fit_fn(model, X_tr, y_tr, X_val, y_val) -> trained model
        Can ignore X_val/y_val if not needed.

    predict_fn : callable
        predict_fn(model, X) -> predictions

    param_space : dict
        Example:
        {
            "hidden_dim": {"type": "int", "low": 32, "high": 256},
            "lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "activation": {"type": "categorical", "choices": ["relu", "tanh"]}
        }

    preprocess_fn : callable or None
        preprocess_fn(X_tr, X_val) -> (X_tr_proc, X_val_proc, preprocess_artifact)
        Optional. Useful if you want fold-specific preprocessing.

    metric_fn : callable
        metric_fn(y_true, y_pred) -> scalar

    direction : str
        "minimize" or "maximize"
    """
    X_train = np.asarray(X_train) if not hasattr(X_train, "iloc") else X_train
    y_train = np.asarray(y_train).reshape(-1)
    X_test = np.asarray(X_test) if not hasattr(X_test, "iloc") else X_test
    y_test = np.asarray(y_test).reshape(-1)

    splits = make_kfold_splits(
        n_samples=len(y_train),
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )

    def objective(trial):
        params = {
            name: suggest_from_space(trial, name, spec)
            for name, spec in param_space.items()
        }

        fold_scores = []

        for train_idx, val_idx in splits:
            if hasattr(X_train, "iloc"):
                X_tr = X_train.iloc[train_idx]
                X_val = X_train.iloc[val_idx]
            else:
                X_tr = X_train[train_idx]
                X_val = X_train[val_idx]

            y_tr = y_train[train_idx]
            y_val = y_train[val_idx]

            if preprocess_fn is not None:
                X_tr_proc, X_val_proc, _ = preprocess_fn(X_tr, X_val)
            else:
                X_tr_proc, X_val_proc = X_tr, X_val

            model = model_builder(params)
            model = fit_fn(model, X_tr_proc, y_tr, X_val_proc, y_val)

            preds = predict_fn(model, X_val_proc)
            score = metric_fn(y_val, preds)
            fold_scores.append(score)

        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        study_name=study_name
    )
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_cv_score = study.best_value

    # Refit best model on full training data
    if preprocess_fn is not None:
        X_train_proc, X_test_proc, preprocess_artifact = preprocess_fn(X_train, X_test)
    else:
        X_train_proc, X_test_proc = X_train, X_test
        preprocess_artifact = None

    best_model = model_builder(best_params)
    best_model = fit_fn(best_model, X_train_proc, y_train, X_test_proc, y_test)

    test_preds = predict_fn(best_model, X_test_proc)

    results = {
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "best_model": best_model,
        "test_rmse": rmse(y_test, test_preds),
        "test_r2": r2_score_manual(y_test, test_preds),
        "test_predictions": test_preds,
        "preprocess_artifact": preprocess_artifact,
    }

    if return_study:
        results["study"] = study

    return results