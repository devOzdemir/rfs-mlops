import os
import requests
from flask import Flask, render_template, request

app = Flask(__name__)

# Docker Network içindeki adresler
API_URL = os.getenv("API_URL", "http://rfs_api:8000/predict")
INFO_URL = os.getenv("API_INFO_URL", "http://rfs_api:8000/info")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    model_info = None  # <-- YENİ
    error = None
    form_data = {}
    options = {}

    # 1. Options Çek (Aynı)
    try:
        resp = requests.get(INFO_URL, timeout=2)
        if resp.status_code == 200:
            options = resp.json()
    except Exception:
        pass

    # 2. POST İşlemi
    if request.method == "POST":
        try:
            # ... (Helper fonksiyonlar ve payload hazırlama AYNI) ...

            # Form verilerini alıyoruz (Aynı kodlar)
            def safe_int(val):
                return int(val) if val and val.strip() else None

            def safe_float(val):
                return float(val) if val and val.strip() else None

            def safe_str(val):
                return val if val and val.strip() else None

            payload = {
                "brand": safe_str(request.form.get("brand")),
                "platform": safe_str(request.form.get("platform")),
                "operating_system": safe_str(request.form.get("operating_system")),
                "panel_type": safe_str(request.form.get("panel_type")),
                "gpu_model": safe_str(request.form.get("gpu_model")),
                "ram_type": safe_str(request.form.get("ram_type")),
                "cpu_family": safe_str(request.form.get("cpu_family")),
                "resolution": request.form.get("resolution"),
                "screen_size_inch": safe_float(request.form.get("screen_size_inch")),
                "ram_gb": safe_int(request.form.get("ram_gb")),
                "ssd_gb": safe_int(request.form.get("ssd_gb")),
                "cpu_generation": safe_int(request.form.get("cpu_generation")),
                "cpu_max_ghz": safe_float(request.form.get("cpu_max_ghz")),
                "gpu_vram_gb": safe_int(request.form.get("gpu_vram_gb")),
                "refresh_rate_hz": safe_int(request.form.get("refresh_rate_hz")),
            }

            form_data = payload  # UI'a geri göndermek için

            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                result = response.json()
                prediction = f"{result['predicted_price_try']:,.2f}"
                # API'den gelen model ismini al
                model_info = result.get("model_info", "Yapay Zeka Modeli")
            else:
                error = f"API Hatası: {response.text}"

        except Exception as e:
            error = f"Sistem Hatası: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        model_info=model_info,  # <-- HTML'e gönderiyoruz
        error=error,
        form_data=form_data,
        options=options,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
