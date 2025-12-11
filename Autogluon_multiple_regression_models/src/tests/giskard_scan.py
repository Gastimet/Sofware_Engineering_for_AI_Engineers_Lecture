import argparse
import pandas as pd
import giskard
from autogluon.tabular import TabularPredictor
import mlflow


def scan_model(model_path, data_path, target_col="churned"):
    # 1. Modeli Yükle
    predictor = TabularPredictor.load(model_path)

    # 2. Veriyi Yükle (Test için küçük bir parça yeterli)
    df = pd.read_csv(data_path).head(20)  # ilk 20 satır

    # Veriyi ve Target'ı ayır
    X = df.drop(columns=[target_col], errors='ignore')
    y = df[target_col] if target_col in df.columns else None

    # 3. Giskard Model ve Dataset Objelerini Oluştur
    giskard_model = giskard.Model(
        model=lambda df: predictor.predict_proba(df).iloc[:, 1].values,  # Binary proba
        model_type="classification",
        feature_names=X.columns.tolist(),
        classification_labels=[0, 1]
    )

    giskard_dataset = giskard.Dataset(
        df=pd.concat([X, y], axis=1),
        target=target_col,
        cat_columns=[]  # Kategorik sütunlarınız varsa buraya yazın
    )

    # 4. Taramayı Başlat
    print("Starting Giskard Scan...")
    scan_results = giskard.scan(giskard_model, giskard_dataset)

    # 5. Raporu Kaydet
    print("Scan complete. Saving report...")
    scan_results.to_html("giskard_report.html")

    # Opsiyonel: Sonuçları konsola yaz
    print(f"Issues found: {len(scan_results.issues)}")

    # Eğer kritik hata varsa process fail etsin (CI/CD'yi durdurmak için)
    if len(scan_results.issues) > 5:  # Örnek eşik değeri
        print("Too many issues found!")
        # exit(1) # Jenkins'i kırmak isterseniz açın


if __name__ == "__main__":
    # Basit kullanım örneği
    # Not: Bu path'ler eğitim sonrası oluşan artifact path'lerine göre düzenlenmeli
    # Örnek olarak hard-coded path verilmiştir.
    pass
    # Gerçek senaryoda bu script'i train.py içinden çağırabilir
    # veya argümanlarla çalıştırabilirsiniz.