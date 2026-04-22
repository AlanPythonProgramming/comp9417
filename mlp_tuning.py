from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def model_builder_keras(params: Dict[str, Any], input_dim: int | None = None):
    """
    Build a Keras MLP regressor from a parameter dictionary.
    """
    if input_dim is None:
        raise ValueError("input_dim must be provided for the Keras model builder.")

    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for _ in range(params["num_layers"]):
        model.add(layers.Dense(params["hidden_dim"], activation=params["activation"]))
        if params["dropout"] > 0:
            model.add(layers.Dropout(params["dropout"]))

    model.add(layers.Dense(1))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params["lr"]),
        loss="mse",
    )
    return model


def make_keras_builder(input_dim: int):
    """
    Return a builder function that only needs params.
    """
    def builder(params: Dict[str, Any]):
        return model_builder_keras(params, input_dim=input_dim)
    return builder


def fit_fn_keras(model, X_tr, y_tr, X_val, y_val):
    """
    Fit Keras model with early stopping.
    """
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=25,
        restore_best_weights=True,
    )

    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        verbose=0,
        callbacks=[early_stop],
    )
    return model


def predict_fn_keras(model, X):
    """
    Predict using Keras model and flatten output.
    """
    return model.predict(X, verbose=0).reshape(-1)


def build_mlp_preprocessor(X_train, target_col: Optional[str] = None) -> ColumnTransformer:
    """
    Build preprocessing pipeline for MLP.

    Uses:
    - median imputation + scaling for numeric columns
    - constant imputation + one-hot encoding for categorical columns
    """
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
                    ("scaler", StandardScaler()),
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


def tune_mlp_regressor(
    X_train,
    y_train,
    X_test,
    y_test,
    target_col: Optional[str] = None,
    n_trials: int = 20,
    n_splits: int = 5,
    param_space: Optional[Dict[str, Dict[str, Any]]] = None,
    random_seed: int = 42,
):
    """
    Preprocess train/test data, tune a Keras MLP using bayes_tune_model,
    and return tuning results together with the fitted preprocessor.
    """
    from hp_script import bayes_tune_model

    if param_space is None:
        param_space = {
            "hidden_dim": {"type": "int", "low": 32, "high": 256},
            "num_layers": {"type": "int", "low": 1, "high": 4},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "activation": {"type": "categorical", "choices": ["relu", "tanh"]},
        }

    # optional reproducibility
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    preprocessor = build_mlp_preprocessor(X_train=X_train, target_col=target_col)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    X_train_processed = np.asarray(X_train_processed, dtype=np.float32)
    X_test_processed = np.asarray(X_test_processed, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    y_test = np.asarray(y_test, dtype=np.float32).reshape(-1)

    builder = make_keras_builder(input_dim=X_train_processed.shape[1])

    results = bayes_tune_model(
        X_train=X_train_processed,
        y_train=y_train,
        X_test=X_test_processed,
        y_test=y_test,
        model_builder=builder,
        fit_fn=fit_fn_keras,
        predict_fn=predict_fn_keras,
        param_space=param_space,
        n_trials=n_trials,
        n_splits=n_splits,
    )

    return results, preprocessor