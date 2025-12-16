from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Docker path ayarı
sys.path.append("/opt/airflow")

from src.rfs.models.train import IndustrialTrainer

# --- AYARLAR (.env Dosyasından Okuma) ---
PROJECT_PATH = os.getenv("PROJECT_PATH")
SSH_CONN_ID = os.getenv("MAC_SSH_NAME", "my_local_mac")
# YENİ: Docker komutunun olduğu dizini .env'den alıyoruz
MAC_EXEC_PATH = os.getenv("MAC_EXEC_PATH")

# Hata Önleme
if not PROJECT_PATH or not MAC_EXEC_PATH:
    raise ValueError(
        "❌ KRİTİK HATA: 'PROJECT_PATH' veya 'MAC_EXEC_PATH' ortam değişkeni bulunamadı! "
        "Lütfen .env dosyanızı kontrol edin."
    )


def run_training_logic():
    """
    IndustrialTrainer sınıfını kullanarak eğitim sürecini yöneten wrapper.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://rfs_mlflow:5000")
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

    print(f"🧠 Eğitim Başlıyor... (MLflow: {tracking_uri})")

    try:
        trainer = IndustrialTrainer()
        winner_model = trainer.run_benchmark()
        print(f"🏆 Kazanan Model: {winner_model}")

        if winner_model:
            trainer.optimize_champion(winner_model)
            print("✅ Şampiyon model optimize edildi ve MLflow'a kaydedildi.")
        else:
            print("⚠️ Uyarı: Benchmark sonucunda uygun bir model bulunamadı.")

    except Exception as e:
        print(f"❌ Eğitim Hatası: {e}")
        raise e


# --- DAG Tanımları ---
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "email_on_failure": False,
    "retries": 0,
}

with DAG(
    "02_rfs_train_api",
    default_args=default_args,
    description="Sadece eğitim yapar ve API servisini restart eder.",
    schedule_interval=None,
    catchup=False,
    tags=["training", "cd", "api-restart"],
) as dag:
    # 1. Görev: Modeli Eğit
    train_task = PythonOperator(
        task_id="train_model",
        python_callable=run_training_logic,
    )

    # 2. Görev: API'yi Restart Et (Dinamik Path ile)
    restart_api_task = SSHOperator(
        task_id="restart_api_container",
        ssh_conn_id=SSH_CONN_ID,
        # Dinamik PATH kullanımı:
        command=f"export PATH=$PATH:{MAC_EXEC_PATH} && cd {PROJECT_PATH} && docker compose restart api",
        cmd_timeout=600,
    )

    train_task >> restart_api_task
