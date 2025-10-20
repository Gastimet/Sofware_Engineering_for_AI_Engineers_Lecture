from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from typing import Optional


class OnlineMeta:
    """
    Base modellerden gelen olasılıkları [p1, p2, ...] feature olarak alıp
    online güncellenen bir logistic meta-öğrenici uygular.
    """

    def __init__(self, state_path: str, n_models: int = 2):
        self.state_path = state_path
        self.n_models = n_models
        self._classes_ = np.array([0, 1])
        self.model: Pipeline = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", SGDClassifier(loss="log_loss", learning_rate="optimal", random_state=42)),
            ]
        )
        self._init = False
        self.load()

    def load(self):
        try:
            obj = joblib.load(self.state_path)
            self.model = obj["model"]
            self._init = True
        except Exception:
            # ilk çalışmada henüz state yok
            pass

    def save(self):
        joblib.dump({"model": self.model}, self.state_path)

    def predict_proba(self, base_probas: np.ndarray) -> float:
        X = np.array(base_probas, dtype=float).reshape(1, -1)
        if not self._init:
            # ilk tahminlerde basit ortalama
            return float(np.clip(np.mean(X), 1e-6, 1 - 1e-6))
        p = self.model.predict_proba(X)[0, 1]
        return float(np.clip(p, 1e-6, 1 - 1e-6))

    def partial_fit(self, base_probas: np.ndarray, y_true: int):
        X = np.array(base_probas, dtype=float).reshape(1, -1)
        y = np.array([y_true])
        if not self._init:
            # ilk kez initialize
            self.model.fit(X, y)
            self._init = True
        else:
            self.model.partial_fit(X, y, classes=self._classes_)
        self.save()
