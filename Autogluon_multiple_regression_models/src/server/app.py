from __future__ import annotations

import threading
import time
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import mlflow
import mlflow.lightgbm
import mlflow.catboost
from mlflow.tracking import MlflowClient
from fastapi import FastAPI
from pydantic import BaseModel

from ..config import settings
from .meta import OnlineMeta

app = FastAPI(title="Multimodel MLflow API", version="1.0")

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
client = MlflowClient()

# Global state
MODELS: Dict[str, Any] = {}
VERSIONS: Dict[str, int] = {}
LOCK = threading.RLock()

META = OnlineMeta(state_path=settings.meta_state_path, n_models=len(settings.model_names))


class PredictRequest(BaseModel):
    features: Dict[str, Any]


class PredictResponse(BaseModel):
    base_probas: Dict[str, float]
    ensemble_proba: float


class FeedbackRequest(BaseModel):
    features: Dict[str, Any]
    y_true: int


def _load_production(name: str):
    try:
        latest = client.get_latest_versions(name, stages=["Production"])
    except Exception:
        return None, None
    if not latest:
        return None, None

    v = int(latest[0].version)
    try:
        if name.lower().endswith("ag"):
            import mlflow.pyfunc
            model = mlflow.pyfunc.load_model(f"models:/{name}/Production")
        elif name.lower().endswith("lgbm"):
            try:
                model = mlflow.sklearn.load_model(f"models:/{name}/Production")
            except Exception:
                model = mlflow.lightgbm.load_model(f"models:/{name}/Production")
        elif name.lower().endswith("cat"):
            model = mlflow.catboost.load_model(f"models:/{name}/Production")
        else:
            try:
                model = mlflow.sklearn.load_model(f"models:/{name}/Production")
            except Exception:
                import mlflow.pyfunc
                model = mlflow.pyfunc.load_model(f"models:/{name}/Production")
    except Exception:
        return None, None

    return model, v


def _poller():
    while True:
        time.sleep(settings.poll_seconds)
        try:
            for name in settings.model_names:
                model, v = _load_production(name)
                if model is None:
                    continue
                with LOCK:
                    if VERSIONS.get(name) != v:
                        MODELS[name] = model
                        VERSIONS[name] = v
                        print(f"[Hot-Reload] {name} -> v{v}")
        except Exception as e:
            print("Poller error:", e)


# İlk yükleme + poller
threading.Thread(target=_poller, daemon=True).start()
for name in settings.model_names:
    try:
        m, v = _load_production(name)
        if m is not None:
            MODELS[name] = m
            VERSIONS[name] = v
    except Exception as e:
        print(f"[Init] {name} yüklenemedi:", e)


@app.get("/health")
def health():
    with LOCK:
        loaded = {k: f"v{VERSIONS.get(k, '-')}" for k in settings.model_names if k in MODELS}
    return {"status": "ok", "loaded_models": loaded}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    x = pd.DataFrame([req.features])
    base_probas: List[float] = []
    base_dict: Dict[str, float] = {}
    with LOCK:
        for name in settings.model_names:
            model = MODELS.get(name)
            if model is None:
                continue
            if hasattr(model, "predict_proba"):
                p = float(model.predict_proba(x)[:, 1][0])
            else:
                pred = model.predict(x)
                # pyfunc (AutoGluon) predict -> olasılık vektörü/seri olabilir
                try:
                    p = float(np.ravel(pred)[0])
                except Exception:
                    p = float(pred[0])
            base_probas.append(p)
            base_dict[name] = p

    if not base_probas:
        return PredictResponse(base_probas={}, ensemble_proba=0.5)

    ens = META.predict_proba(np.array(base_probas))
    return PredictResponse(base_probas=base_dict, ensemble_proba=ens)


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    x = pd.DataFrame([req.features])
    base_probas: List[float] = []
    with LOCK:
        for name in settings.model_names:
            model = MODELS.get(name)
            if model is None:
                continue
            if hasattr(model, "predict_proba"):
                p = float(model.predict_proba(x)[:, 1][0])
            else:
                pred = model.predict(x)
                try:
                    p = float(np.ravel(pred)[0])
                except Exception:
                    p = float(pred[0])
            base_probas.append(p)

    if base_probas:
        META.partial_fit(np.array(base_probas), int(req.y_true))

    try:
        with mlflow.start_run(run_name="online_feedback", nested=True):
            mlflow.log_metric("y_true", int(req.y_true))
            for i, p in enumerate(base_probas):
                mlflow.log_metric(f"base_p_{i}", float(p))
    except Exception:
        pass

    return {"status": "updated"}