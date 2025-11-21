from __future__ import annotations

import threading
import time
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import mlflow
import mlflow.lightgbm
import mlflow.catboost
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from ..config import settings
from .meta import OnlineMeta

app = FastAPI(title="Multimodel MLflow API (Secured)", version="1.0")

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
client = MlflowClient()

# Global state
MODELS: Dict[str, Any] = {}
VERSIONS: Dict[str, int] = {}
LOCK = threading.RLock()

META = OnlineMeta(state_path=settings.meta_state_path, n_models=len(settings.model_names))


# --- MLSecOps: Strict Input Schema Definition ---
class ChurnFeatures(BaseModel):
    """
    Strict schema for the churn prediction task.
    Defines exact data types and constraints to prevent bad data injection.
    """
    age: int
    income: float
    tenure: int
    tx_count: int

    # Validator to prevent nonsense age values
    @field_validator('age')
    @classmethod
    def check_age(cls, v: int) -> int:
        if v < 18 or v > 120:
            raise ValueError('Age must be between 18 and 120')
        return v

    # Validator to ensure non-negative financial/time values
    @field_validator('income', 'tenure', 'tx_count')
    @classmethod
    def check_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError('Value must be non-negative')
        return v


class PredictRequest(BaseModel):
    # Instead of Dict[str, Any], we use the strict schema
    features: ChurnFeatures


class PredictResponse(BaseModel):
    base_probas: Dict[str, float]
    ensemble_proba: float


class FeedbackRequest(BaseModel):
    features: ChurnFeatures
    y_true: int

    @field_validator('y_true')
    @classmethod
    def check_binary(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError('Target must be 0 or 1')
        return v


# -----------------------------------------------


def _load_production(name: str):
    try:
        latest = client.get_latest_versions(name, stages=["Production"])
    except Exception:
        return None, None
    if not latest:
        return None, None

    v = int(latest[0].version)
    try:
        # Safe model loading based on naming convention
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
    except Exception as e:
        print(f"[Error] Failed to load model {name}: {e}")
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


# Initial load + start poller
threading.Thread(target=_poller, daemon=True).start()
for name in settings.model_names:
    try:
        m, v = _load_production(name)
        if m is not None:
            MODELS[name] = m
            VERSIONS[name] = v
    except Exception as e:
        print(f"[Init] {name} could not be loaded:", e)


@app.get("/health")
def health():
    with LOCK:
        loaded = {k: f"v{VERSIONS.get(k, '-')}" for k in settings.model_names if k in MODELS}
    return {"status": "ok", "loaded_models": loaded}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Securely convert validated Pydantic model to DataFrame
    # model_dump() ensures we only get the fields defined in ChurnFeatures
    input_data = req.features.model_dump()
    x = pd.DataFrame([input_data])

    base_probas: List[float] = []
    base_dict: Dict[str, float] = {}

    with LOCK:
        for name in settings.model_names:
            model = MODELS.get(name)
            if model is None:
                continue

            try:
                if hasattr(model, "predict_proba"):
                    p = float(model.predict_proba(x)[:, 1][0])
                else:
                    pred = model.predict(x)
                    # pyfunc (AutoGluon) predict -> might be probability vector or series
                    try:
                        p = float(np.ravel(pred)[0])
                    except Exception:
                        p = float(pred[0])
                base_probas.append(p)
                base_dict[name] = p
            except Exception as e:
                print(f"Prediction error in {name}: {e}")
                # Fallback or skip could be handled here
                continue

    if not base_probas:
        # Fail safely if no models are available
        return PredictResponse(base_probas={}, ensemble_proba=0.5)

    ens = META.predict_proba(np.array(base_probas))
    return PredictResponse(base_probas=base_dict, ensemble_proba=ens)


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    # Secure conversion
    input_data = req.features.model_dump()
    x = pd.DataFrame([input_data])

    base_probas: List[float] = []
    with LOCK:
        for name in settings.model_names:
            model = MODELS.get(name)
            if model is None:
                continue
            try:
                if hasattr(model, "predict_proba"):
                    p = float(model.predict_proba(x)[:, 1][0])
                else:
                    pred = model.predict(x)
                    try:
                        p = float(np.ravel(pred)[0])
                    except Exception:
                        p = float(pred[0])
                base_probas.append(p)
            except Exception:
                continue

    if base_probas:
        META.partial_fit(np.array(base_probas), req.y_true)

    # Log feedback to MLflow safely
    try:
        with mlflow.start_run(run_name="online_feedback", nested=True):
            mlflow.log_metric("y_true", req.y_true)
            for i, p in enumerate(base_probas):
                mlflow.log_metric(f"base_p_{i}", float(p))
    except Exception as e:
        print(f"Logging error: {e}")

    return {"status": "updated"}