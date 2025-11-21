from __future__ import annotations
import argparse
import os
import tempfile
from typing import List

import numpy as np
import pandas as pd
import polars as pl

import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.catboost
import mlflow.pyfunc

from .trainers import train_lightgbm, train_catboost
from ..data.connectors import CSVConnector
from ..data.merge import join_many


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="churn")
    p.add_argument("--data-csv-a", type=str, required=True)
    p.add_argument("--data-csv-b", type=str, required=True)
    p.add_argument("--join-keys", type=str, default="customer_id")
    p.add_argument("--target", type=str, required=True)
    p.add_argument("--promote", action="store_true")
    return p.parse_args()


class AGPyfunc(mlflow.pyfunc.PythonModel):
    """AutoGluon TabularPredictor'ü pyfunc olarak sarar; predict() olasılık döndürür."""

    def load_context(self, context):
        from autogluon.tabular import TabularPredictor
        self.predictor = TabularPredictor.load(context.artifacts["predictor"])
        # pozitif sınıf label'ını bul (binary)
        try:
            classes = list(self.predictor.class_labels)
            self.pos_label = 1 if 1 in classes else classes[-1]
        except Exception:
            self.pos_label = None

    def predict(self, context, model_input):
        X = pd.DataFrame(model_input)
        try:
            proba = self.predictor.predict_proba(X)
            if self.pos_label is not None and self.pos_label in proba.columns:
                return proba[self.pos_label].values
            return proba.iloc[:, -1].values
        except Exception:
            preds = self.predictor.predict(X)
            return (preds == 1).astype(float).values


def main():
    args = parse_args()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(f"{args.task}_experiment")

    # 1) veri yükle & join
    df_a = CSVConnector(args.data_csv_a).load()
    df_b = CSVConnector(args.data_csv_b).load()
    join_keys: List[str] = [k.strip() for k in args.join_keys.split(",")]
    df = join_many([df_a, df_b], on=join_keys)

    # 2) y (writable) + X (pandas, Arrow kapalı)
    assert args.target in df.columns, f"Target {args.target} not found"
    y_np = df[args.target].cast(pl.Int64).to_numpy()
    y = pd.Series(y_np, name=args.target).astype("int64", copy=True)

    # FIX: Removed 'strict=False' because Polars 0.20.0 does not support it.
    X = df.drop(join_keys + [args.target]).to_pandas(use_pyarrow_extension_array=False).copy()

    # ---------- LightGBM ----------
    with mlflow.start_run(run_name="lightgbm"):
        mlflow.sklearn.autolog(log_models=True)
        lgbm, m1 = train_lightgbm(X, y)
        mlflow.log_metrics(m1)
        mlflow.lightgbm.log_model(lgbm, artifact_path="model")
        result = mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
            name="model_lgbm",
        )
        v_lgbm = int(result.version)

    # ---------- CatBoost ----------
    with mlflow.start_run(run_name="catboost"):
        mlflow.sklearn.autolog(log_models=True)
        cat, m2 = train_catboost(X, y)
        mlflow.log_metrics(m2)
        mlflow.catboost.log_model(cat, artifact_path="model")
        result = mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
            name="model_cat",
        )
        v_cat = int(result.version)

    # ---------- AutoGluon (pyfunc) ----------
    with mlflow.start_run(run_name="autogluon"):
        from autogluon.tabular import TabularPredictor

        train_df = pd.concat([X, y], axis=1)
        predictor = TabularPredictor(
            label=args.target,
            problem_type="binary",
            eval_metric="log_loss"
        ).fit(
            train_data=train_df,
            presets="medium_quality_faster_train",
            hyperparameters={
                "GBM": {},  # LightGBM
                "RF": {},  # RandomForest
                "XT": {},  # ExtraTrees
                "KNN": {},  # hızlı seçenekler
                # ağır olanlar kapalı: "CAT": {}, "XGB": {}, "NN_TORCH": {}
            },
            verbosity=2,
        )

        # basit metrik (train üstünde logloss)
        try:
            from sklearn.metrics import log_loss
            p = predictor.predict_proba(X)
            pp = p[1].values if 1 in p.columns else p.iloc[:, -1].values
            mlflow.log_metric("logloss", float(log_loss(y, np.clip(pp, 1e-6, 1 - 1e-6), labels=[0, 1])))
        except Exception:
            pass

        # pyfunc olarak logla ve kaydet
        with tempfile.TemporaryDirectory() as td:
            save_dir = os.path.join(td, "ag_model")
            predictor.save(save_dir)
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=AGPyfunc(),
                artifacts={"predictor": save_dir},
                pip_requirements=[
                    "autogluon.tabular==1.1.1",
                    "numpy==1.26.4",
                    "pandas==2.2.2",
                    "scikit-learn==1.4.0",
                    "lightgbm==4.3.0",
                ],
            )
            result = mlflow.register_model(
                model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
                name="model_ag",
            )
            v_ag = int(result.version)

    print("LightGBM:", m1)
    print("CatBoost:", m2)
    print("AutoGluon: logged & registered")

    if args.promote:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        client.transition_model_version_stage(name="model_lgbm", version=v_lgbm, stage="Production",
                                              archive_existing_versions=True)
        client.transition_model_version_stage(name="model_cat", version=v_cat, stage="Production",
                                              archive_existing_versions=True)
        client.transition_model_version_stage(name="model_ag", version=v_ag, stage="Production",
                                              archive_existing_versions=True)
        print("Promoted to Production:", f"model_lgbm v{v_lgbm}, model_cat v{v_cat}, model_ag v{v_ag}")


if __name__ == "__main__":
    main()