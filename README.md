# Multimodel MLflow Sistemi


- Çoklu veri kaynağı (CSV/DB/REST) -> birleştirme (Polars)
- Multimodel (LightGBM + CatBoost) -> Ensemble (online meta logistic)
- MLflow Tracking + Model Registry + Autolog
- FastAPI servis: runtime hot-reload + online öğrenme


## Hızlı Başlangıç
1) docker compose up -d --build
2) docker compose exec api python -m src.pipeline.train --task churn --data-csv-a example_data/source_a.csv --data-csv-b example_data/source_b.csv --target churned --promote
3) API: http://localhost:8000/docs, MLflow: http://localhost:5000