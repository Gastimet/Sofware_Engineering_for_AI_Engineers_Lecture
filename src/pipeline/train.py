from __future__ import annotations
import argparse
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.catboost
import pandas as pd
import polars as pl
import numpy as np
from typing import List
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


def main():
    args = parse_args()

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(f"{args.task}_experiment")

    # 1) Load & join
    df_a = CSVConnector(args.data_csv_a).load()
    df_b = CSVConnector(args.data_csv_b).load()
    join_keys: List[str] = [k.strip() for k in args.join_keys.split(",")]
    df = join_many([df_a, df_b], on=join_keys)

    # 2) y writable + X pandas (Arrow kapalı)
    assert args.target in df.columns, f"Target {args.target} not found"
    y_np = df[args.target].cast(pl.Int64).to_numpy()
    y = pd.Series(y_np, name=args.target).astype("int64", copy=True)
    X = df.drop(join_keys + [args.target], strict=False).to_pandas(use_pyarrow_extension_array=False).copy()

    # 3) Train & log: LGBM (tabular only)
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

    # 4) Train & log: CatBoost (tabular only)
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

    print("LightGBM:", m1)
    print("CatBoost:", m2)

    if args.promote:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        client.transition_model_version_stage(name="model_lgbm", version=v_lgbm, stage="Production", archive_existing_versions=True)
        client.transition_model_version_stage(name="model_cat", version=v_cat, stage="Production", archive_existing_versions=True)
        print("Promoted to Production:", f"model_lgbm v{v_lgbm}, model_cat v{v_cat}")


if __name__ == "__main__":
    main()
