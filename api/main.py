import os
import json
import math
import numpy as np
import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from api.schemas import LaptopInput

# Global değişkenler
model_pipeline = None
dropdown_options = {}
model_metadata = {
    "name": "Unknown",
    "version": "0",
}  # <-- YENİ: Model bilgisini tutacak


def load_champion_assets():
    global model_pipeline, dropdown_options, model_metadata
    try:
        print("🔌 MLflow'a bağlanılıyor...")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://rfs_mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)

        # 1. Model Bilgilerini ve Kendisini Al
        model_name = "RFS_Laptop_Price_Predictor"
        model_alias = "champion"

        # Versiyon bilgisini MLflow'dan çek
        client = mlflow.tracking.MlflowClient()
        model_version_info = client.get_model_version_by_alias(model_name, model_alias)

        # Metadata'yı güncelle
        model_metadata = {
            "name": model_name,
            "version": model_version_info.version,
            "run_id": model_version_info.run_id,
        }

        # Modeli yükle
        model_uri = f"models:/{model_name}@{model_alias}"
        print(f"📥 Model indiriliyor: {model_uri} (v{model_metadata['version']})")
        model_pipeline = mlflow.sklearn.load_model(model_uri)

        # 2. Seçenekler JSON'ını İndir
        run_id = model_version_info.run_id
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="categorical_options.json"
        )

        with open(local_path, "r", encoding="utf-8") as f:
            dropdown_options = json.load(f)

        print("✅ Yükleme Tamamlandı!")

    except Exception as e:
        print(f"⚠️ Yükleme hatası: {e}")
        dropdown_options = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_champion_assets()
    yield
    print("🛑 API Kapanıyor...")


app = FastAPI(title="RFS API", version="1.0", lifespan=lifespan)


# ... (Health ve Info endpointleri AYNI kalacak) ...
@app.get("/health")
def health_check():
    if model_pipeline is None:
        return {"status": "unhealthy"}
    return {"status": "healthy", "model": model_metadata}


@app.get("/info")
def get_options():
    return dropdown_options


@app.post("/predict")
def predict_price(input_data: LaptopInput):
    if not model_pipeline:
        raise HTTPException(status_code=503, detail="Model hizmete hazır değil.")

    # ... (Veri işleme ve PPI hesaplama kısımları AYNI kalsın) ...
    data_dict = input_data.model_dump()
    res_str = data_dict.pop("resolution")
    try:
        w, h = map(int, res_str.lower().split("x"))
        inches = data_dict.get("screen_size_inch", 15.6)
        ppi = math.sqrt(w**2 + h**2) / inches
        data_dict["ppi"] = ppi
    except Exception:
        data_dict["ppi"] = np.nan

    df = pd.DataFrame([data_dict])
    df.fillna(value=np.nan, inplace=True)

    try:
        prediction = model_pipeline.predict(df)

        # CEVAP JSON'INI GÜNCELLİYORUZ
        return {
            "predicted_price_try": round(float(prediction[0]), 2),
            "currency": "TRY",
            "model_info": f"{model_metadata['name']} (v{model_metadata['version']})",  # <-- YENİ
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
