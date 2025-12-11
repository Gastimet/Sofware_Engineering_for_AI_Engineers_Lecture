import os
import shutil
import pandas as pd
import giskard
import mlflow
from autogluon.tabular import TabularPredictor

# Sabitler
MLFLOW_TRACKING_URI = "http://mlflow:5000"
DATA_PATH = "/workspace/Autogluon_multiple_regression_models/example_data/source_a.csv"
TARGET_COL = "churned"
REPORT_OUTPUT = "giskard_report.html"


def get_latest_model_path():
    """MLflow'dan en son başarılı 'churn_experiment' koşusunu bulur ve modeli indirir."""
    print(f"📡 MLflow'a bağlanılıyor: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    experiment = mlflow.get_experiment_by_name("churn_experiment")
    if experiment is None:
        raise ValueError("❌ 'churn_experiment' bulunamadı. Önce eğitimi çalıştırdığınızdan emin olun.")

    # En son çalışan run'ı bul
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1
    )

    if runs.empty:
        raise ValueError("❌ Hiçbir başarılı 'run' bulunamadı.")

    last_run_id = runs.iloc[0].run_id
    print(f"✅ En son Run ID bulundu: {last_run_id}")

    # AutoGluon model dosyaları MLflow içinde 'model/artifacts/predictor' altında saklanıyor
    # train.py içindeki log_model yapısına göre artifact path'i belirliyoruz
    artifact_uri = f"runs:/{last_run_id}/model/artifacts/predictor"

    print(f"📥 Model indiriliyor: {artifact_uri}")
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
    return local_path


def scan():
    try:
        # 1. Modeli MLflow'dan Çek
        model_path = get_latest_model_path()
        predictor = TabularPredictor.load(model_path)
        print("✅ Model başarıyla yüklendi.")

        # 2. Test Verisini Hazırla
        # Gerçek bir test için verinin bir kısmını (veya validation setini) kullanıyoruz.
        print(f"📂 Veri seti yükleniyor: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH).head(500)  # Hız için ilk 500 satır yeterli

        # Giskard için target sütununu ayırın ama dataset objesinde tutun
        # Veri hazırlığı (train.py'daki mantığa benzer olmalı)
        # Eğer eğitimde 'customer_id' gibi kolonlar atıldıysa burada da dikkat edilmeli,
        # ancak Giskard Dataset objesi ham veriyi sever, modeli sararken feature'ları seçeriz.

        # 3. Giskard Model Wrapper Oluştur
        # AutoGluon'un predict_proba çıktısını Giskard'ın formatına uygun hale getiren fonksiyon
        def prediction_function(df):
            # AutoGluon DataFrame bekler
            res = predictor.predict_proba(df)
            # Binary classification için pozitif sınıfın (1) olasılığını döndürelim
            # Eğer output [0, 1] kolonlarına sahipse 1'i al
            if 1 in res.columns:
                return res[1].values
            else:
                return res.iloc[:, -1].values

        giskard_model = giskard.Model(
            model=prediction_function,
            model_type="classification",
            name="Churn Prediction Model",
            feature_names=predictor.feature_metadata_in.get_features(),
            classification_labels=[0, 1]
        )

        # 4. Giskard Dataset Oluştur
        giskard_dataset = giskard.Dataset(
            df=df,
            target=TARGET_COL,
            name="Churn Validation Data",
            cat_columns=df.select_dtypes(include=['object', 'category']).columns.tolist()
        )

        # 5. Taramayı (Scan) Başlat
        print("🕵️ Giskard Taraması Başlatılıyor... (Bu işlem biraz sürebilir)")
        scan_results = giskard.scan(giskard_model, giskard_dataset)

        # 6. Raporu Kaydet
        print(f"📝 Rapor kaydediliyor: {REPORT_OUTPUT}")
        scan_results.to_html(REPORT_OUTPUT)

        # 7. Sonuçları Özetle
        issues = len(scan_results.issues)
        print(f"⚠️ Toplam Tespit Edilen Sorun Sayısı: {issues}")

        if issues > 0:
            print("🔍 Tespit edilen bazı sorunlar:")
            for issue in scan_results.issues[:3]:  # İlk 3 sorunu göster
                print(f" - {issue.meta.name}: {issue.description}")

        print("✅ Giskard Süreci Tamamlandı.")

    except Exception as e:
        print(f"❌ Giskard taraması sırasında hata oluştu: {e}")
        # Hata durumunda pipeline'ı kırmamak için boş rapor oluştur (Opsiyonel: raise e yaparak kırabilirsiniz)
        with open(REPORT_OUTPUT, "w") as f:
            f.write(f"<html><body><h1>Scan Failed</h1><p>{e}</p></body></html>")
        # CI/CD'nin fail olmasını isterseniz aşağıdaki satırı açın:
        # raise e


if __name__ == "__main__":
    scan()