import yaml
import logging.config
import os
from dotenv import load_dotenv  # python-dotenv kütüphanesi


def load_config(config_path="configs/model_config.yaml"):
    """YAML dosyasını okur ve dict olarak döndürür."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_env_and_logging(config_path="configs/logging.yaml"):
    """
    1. .env dosyasını yükler.
    2. MinIO (S3) ayarlarını global AWS ayarlarının önüne geçirir.
    3. Loglamayı başlatır.
    """
    # 1. .env Yükle
    load_dotenv()

    # 2. MinIO Ayarlarını Zorla (Mac'teki local AWS keylerini ezmek için)
    # Boto3 kütüphanesi 'AWS_ACCESS_KEY_ID' arar, bizde 'MINIO_ACCESS_KEY' var. Eşliyoruz.
    if os.getenv("MINIO_ACCESS_KEY"):
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MINIO_ACCESS_KEY")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MINIO_SECRET_KEY")
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

    # 3. Logging Başlat
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.warning(f"Logging config not found at {config_path}, using default.")
