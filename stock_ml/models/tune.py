"""
Hyperparameter tuning using Optuna for XGBoost and LightGBM.
"""
import optuna
import numpy as np
import logging
import os
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from config import MODEL_DIR, RANDOM_STATE, N_SPLITS

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_xgboost(X, y, n_trials: int = 50) -> dict:
    """Tune XGBoost hyperparameters using Optuna. Returns best params."""
    import xgboost as xgb

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 1),
            "random_state": RANDOM_STATE,
            "eval_metric": "logloss",
            "use_label_encoder": False,
        }
        model = xgb.XGBClassifier(**params)
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)
            model.fit(X_tr_s, y_tr)
            preds = model.predict(X_val_s)
            scores.append(f1_score(y_val, preds, zero_division=0))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"XGBoost best F1: {study.best_value:.4f} | params: {study.best_params}")
    return study.best_params


def tune_lightgbm(X, y, n_trials: int = 50) -> dict:
    """Tune LightGBM hyperparameters using Optuna. Returns best params."""
    import lightgbm as lgb

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "random_state": RANDOM_STATE,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)
            model.fit(X_tr_s, y_tr)
            preds = model.predict(X_val_s)
            scores.append(f1_score(y_val, preds, zero_division=0))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"LightGBM best F1: {study.best_value:.4f} | params: {study.best_params}")
    return study.best_params


def tune_and_retrain(ticker: str, label_version: str = "A", n_trials: int = 50):
    """
    Run Optuna tuning for XGBoost and LightGBM, retrain with best params,
    save tuned models to MODEL_DIR with suffix _tuned.
    """
    from features.pipeline import build_feature_matrix
    X, y = build_feature_matrix(ticker, label_version)

    logger.info(f"Starting hyperparameter tuning for {ticker} | label_version={label_version}")

    results = {}

    for model_name, tune_fn in [("xgboost", tune_xgboost), ("lightgbm", tune_lightgbm)]:
        logger.info(f"Tuning {model_name}...")
        best_params = tune_fn(X, y, n_trials=n_trials)

        # Retrain with best params on full dataset
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if model_name == "xgboost":
            import xgboost as xgb
            best_params["random_state"] = RANDOM_STATE
            best_params["eval_metric"] = "logloss"
            best_params["use_label_encoder"] = False
            model = xgb.XGBClassifier(**best_params)
            model.fit(X_scaled, y)
            model_path = os.path.join(MODEL_DIR, f"xgboost_{ticker}_{label_version}_tuned.json")
            model.save_model(model_path)
        else:
            import lightgbm as lgb
            best_params["random_state"] = RANDOM_STATE
            best_params["verbose"] = -1
            model = lgb.LGBMClassifier(**best_params)
            model.fit(X_scaled, y)
            model_path = os.path.join(MODEL_DIR, f"lightgbm_{ticker}_{label_version}_tuned.txt")
            model.booster_.save_model(model_path)

        # Save scaler for tuned model
        scaler_path = os.path.join(MODEL_DIR, f"scaler_{ticker}_{label_version}_tuned.joblib")
        joblib.dump(scaler, scaler_path)

        logger.info(f"Tuned {model_name} saved to {model_path}")
        results[model_name] = {"best_params": best_params, "model_path": model_path}

    return results
