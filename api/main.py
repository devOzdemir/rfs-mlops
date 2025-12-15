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


def load_champion_model():
    """MLflow'dan @champion etiketli modeli indirir."""
    try:
        print("🔌 MLflow'a bağlanılıyor...")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://rfs_mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)

        model_uri = "models:/RFS_Laptop_Price_Predictor@champion"
        print(f"📥 Model indiriliyor: {model_uri}")

        # Modeli yükle
        loaded_model = mlflow.sklearn.load_model(model_uri)
        print("✅ Model başarıyla hafızaya yüklendi!")
        return loaded_model
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model_pipeline
    model_pipeline = load_champion_model()
    yield
    # Shutdown
    print("🛑 API Kapanıyor...")


app = FastAPI(title="RFS Laptop Price Prediction API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health_check():
    if model_pipeline is None:
        return {"status": "unhealthy", "detail": "Model not loaded"}
    return {"status": "healthy", "model": "champion"}


@app.get("/info")
def get_options():
    """Frontend dropdownları için statik listeler (Geliştirilebilir)"""
    return {
        "brands": ["Asus", "Lenovo", "HP", "MSI", "Apple", "Dell", "Acer", "Monster"],
        "operating_systems": [
            "Windows 11 Home",
            "Windows 11 Pro",
            "FreeDOS",
            "macOS",
            "Linux",
        ],
        "cpu_families": [
            "Core i3",
            "Core i5",
            "Core i7",
            "Core i9",
            "Ryzen 3",
            "Ryzen 5",
            "Ryzen 7",
            "Ryzen 9",
            "M1",
            "M2",
            "M3",
        ],
        "gpu_models": [
            "RTX 4050",
            "RTX 4060",
            "RTX 4070",
            "RTX 3050",
            "RTX 3060",
            "Integrated",
        ],
        "panel_types": ["IPS", "OLED", "TN", "VA"],
    }


@app.post("/predict")
def predict_price(input_data: LaptopInput):
    if not model_pipeline:
        raise HTTPException(status_code=503, detail="Model hizmete hazır değil.")

    try:
        # 1. Pydantic verisini Dictionary'e çevir
        data_dict = input_data.model_dump()

        # 2. FEATURE ENGINEERING: PPI Hesapla
        # Kullanıcıdan 'resolution' ve 'screen_size_inch' aldık.
        # Bunlardan 'ppi' türetip, 'resolution'ı sileceğiz.
        res_str = data_dict.pop("resolution")  # Listeden çıkar ve al

        try:
            # "1920x1080" stringini parçala
            w, h = map(int, res_str.lower().split("x"))
            inches = data_dict.get("screen_size_inch", 15.6)

            # PPI Formülü
            ppi = math.sqrt(w**2 + h**2) / inches
            data_dict["ppi"] = ppi

        except Exception:
            # Eğer hesaplanamazsa NaN ver (Imputer doldursun)
            data_dict["ppi"] = np.nan

        # 3. DataFrame Oluştur ve Hazırla
        df = pd.DataFrame([data_dict])

        # Pydantic'ten gelen None değerlerini NumPy NaN yap
        # (Scikit-Learn Imputer, None'ı her zaman tanımaz, NaN sever)
        df.fillna(value=np.nan, inplace=True)

        # 4. Tahmin Yap
        prediction = model_pipeline.predict(df)
        predicted_price = float(prediction[0])

        return {
            "predicted_price_try": round(predicted_price, 2),
            "currency": "TRY",
            "debug_info": {
                "calculated_ppi": round(data_dict["ppi"], 2)
                if not np.isnan(data_dict["ppi"])
                else None
            },
        }

    except Exception as e:
        import traceback

        traceback.print_exc()  # Loglara detaylı hata bas
        raise HTTPException(status_code=500, detail=f"Tahmin motoru hatası: {str(e)}")
