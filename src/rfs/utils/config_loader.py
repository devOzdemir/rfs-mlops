# src/rfs/utils/config_loader.py

import yaml
import logging.config
import os
from dotenv import load_dotenv


def load_config(config_path="configs/model_config.yaml"):
    # ... (Burası aynı kalsın) ...
    if not os.path.exists(config_path):
        # Docker içinde bazen path sorunu olabilir, tam yol deneyelim
        config_path = os.path.join(os.getcwd(), config_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_env_and_logging(config_path="configs/logging.yaml"):
    # 1. .env Yükle
    load_dotenv()

    # 2. MinIO / AWS Ayarları
    # Docker içinden "rfs_minio", dışarıdan "localhost" adresini kullanabilmek için:
    if os.getenv("MINIO_ACCESS_KEY"):
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MINIO_ACCESS_KEY")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MINIO_SECRET_KEY")

        # --- DÜZELTME BURADA ---
        # Eğer sistemde (Docker compose'da) tanımlı bir URL varsa onu al, yoksa localhost yap.
        default_minio = "http://localhost:9000"
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
            "MLFLOW_S3_ENDPOINT_URL", default_minio
        )
        # -----------------------

    # 3. Logging Başlat
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        # Logging dosyası bulunamazsa basit loglama aç
        logging.basicConfig(level=logging.INFO)
        # Uyarı verelim ama süreci durdurmayalım
        print(f"Logging config not found at {config_path}, using default basicConfig.")
