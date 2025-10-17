from __future__ import annotations
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from catboost import CatBoostClassifier


def _safe_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Tiny/imbalanced veri için güvenli bölme."""
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        # Tek sınıf: tüm veri train, test boş
        return X, X.iloc[0:0], y, y.iloc[0:0]

    min_count = counts.min()
    if min_count < 2 or len(y) < 10:
        # Azınlık sınıfı tamamen train'de tut, testi çoğunluktan seç
        minority = classes[np.argmin(counts)]
        majority_idx = y.index[y != minority]

        rng = np.random.RandomState(random_state)
        # En az 2 örnekli test seti hedefle (sınıf sayısı kadar)
        target_test = max(2, int(round(len(y) * test_size)))
        target_test = min(target_test, max(0, len(majority_idx) - 1))

        if target_test > 0:
            test_idx = pd.Index(rng.choice(majority_idx, size=target_test, replace=False))
        else:
            test_idx = pd.Index([])

        train_idx = y.index.difference(test_idx)
        return X.loc[train_idx], X.loc[test_idx], y.loc[train_idx], y.loc[test_idx]

    # Normal durumda stratified split
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def _safe_metrics(y_true: pd.Series, proba: np.ndarray) -> Dict:
    metrics: Dict = {}
    y_true = pd.Series(y_true).astype(int)
    p = np.clip(proba, 1e-6, 1 - 1e-6)

    if y_true.nunique() >= 2 and len(y_true) >= 2:
        metrics["auc"] = float(roc_auc_score(y_true, p))
    else:
        metrics["auc"] = float("nan")

    try:
        metrics["logloss"] = float(log_loss(y_true, p, labels=[0, 1]))
    except Exception:
        metrics["logloss"] = float("nan")

    return metrics


def train_lightgbm(X: pd.DataFrame, y: pd.Series) -> Tuple[lgb.LGBMClassifier, Dict]:
    X_tr, X_te, y_tr, y_te = _safe_split(X, y, test_size=0.2, random_state=42)
    if pd.Series(y_tr).nunique() < 2:
        X_tr, y_tr = X, pd.Series(y).astype(int)
        X_te, y_te = X.iloc[0:0], pd.Series([], dtype=int)

    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=-1,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1] if len(X_te) else np.array([])
    metrics = _safe_metrics(y_te, proba) if len(X_te) else {"auc": float("nan"), "logloss": float("nan")}
    return model, metrics


def train_catboost(X: pd.DataFrame, y: pd.Series) -> Tuple[CatBoostClassifier, Dict]:
    X_tr, X_te, y_tr, y_te = _safe_split(X, y, test_size=0.2, random_state=42)
    if pd.Series(y_tr).nunique() < 2:
        X_tr, y_tr = X, pd.Series(y).astype(int)
        X_te, y_te = X.iloc[0:0], pd.Series([], dtype=int)

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        verbose=False,
        random_seed=42,
    )
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1] if len(X_te) else np.array([])
    metrics = _safe_metrics(y_te, proba) if len(X_te) else {"auc": float("nan"), "logloss": float("nan")}
    return model, metrics
