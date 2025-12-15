# Hafif bir Python imajı kullanıyoruz
FROM python:3.10-slim

WORKDIR /app

# Gerekli sistem paketleri (Postgres driver için libpq-dev gerekebilir)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Python kütüphanelerini yükle
# Not: Modeli yüklemek için 'mlflow', 'boto3' (S3 için) ve modelin kütüphaneleri (sklearn, xgboost vb.) şarttır.
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pandas \
    scikit-learn \
    xgboost \
    lightgbm \
    mlflow \
    boto3 \
    psycopg2-binary \
    pydantic

# Uygulama kodlarını kopyala
# (Docker Compose'da context root olacağı için burayı basit tutuyoruz)
COPY api /app/api
COPY src /app/src 
# src klasörünü de kopyalıyoruz çünkü bazen pickle dosyaları custom modüllere ihtiyaç duyabilir.

# API'yi başlat
# Host 0.0.0.0 olmalı ki dışarıdan erişilebilsin
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]