from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.models.param import Param
from datetime import datetime, timedelta
import sys
import os

# --- KRİTİK: Airflow'un 'src' klasörünü görmesi için ---
sys.path.append("/opt/airflow")

# --- İMPORTLAR ---
# 1. ETL Modülü
from src.rfs.features.pipeline import LaptopETLPipeline

# 2. Training Modülü
from src.rfs.models.train import IndustrialTrainer

# --- AYARLAR (Host Makine) ---
VENV_PYTHON_PATH = "/Users/erwin/Developer/ml-dl/rfs-mlops/.venv/bin/python"
PROJECT_PATH = "/Users/erwin/Developer/ml-dl/rfs-mlops"
SCRIPT_NAME = "run_scraping.py"

default_args = {
    "owner": "rfs_team",
    "retries": 0,
    "retry_delay": timedelta(minutes=2),
}

# --- FONKSİYONLAR ---


def run_project_etl():
    """
    Ham veriyi temizler ve 'features' tablosuna yazar.
    """
    print("🚀 Airflow ETL Başlatılıyor...")
    try:
        pipeline = LaptopETLPipeline()
        pipeline.run()
        print("✅ ETL Başarıyla Tamamlandı!")
    except Exception as e:
        print(f"❌ ETL Sırasında Hata: {e}")
        raise e


def run_model_training():
    """
    (YENİ) Temiz veriyi alır, modelleri yarıştırır ve şampiyonu MLflow'a kaydeder.
    """
    print("🧠 Model Eğitimi Başlıyor...")

    # Docker içinde MLflow tracking URI'yı set edelim
    # Bu ayar train.py içindeki 'localhost' ayarını ezer.
    os.environ["MLFLOW_TRACKING_URI"] = "http://rfs_mlflow:5000"

    try:
        trainer = IndustrialTrainer()

        # 1. Benchmark (Tüm modelleri yarıştır)
        winner_model = trainer.run_benchmark()
        print(f"🏆 Kazanan Model: {winner_model}")

        # 2. Optimizasyon (Kazananı eğit)
        if winner_model:
            trainer.optimize_champion(winner_model)
            print("✅ Şampiyon model optimize edildi ve MLflow'a kaydedildi.")
        else:
            print("⚠️ Benchmark sonucunda uygun model bulunamadı.")

    except Exception as e:
        print(f"❌ Eğitim Hatası: {e}")
        raise e


# --- DAG TANIMI ---

with DAG(
    dag_id="01_rfs_hybrid_pipeline",
    default_args=default_args,
    description="Host(Scraping) -> Docker(ETL) -> Docker(Training)",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["hybrid", "ssh", "scraper", "etl", "training"],
    params={
        "hb_url": Param(
            "https://www.hepsiburada.com/laptop-notebook-dizustu-bilgisayarlar-c-98?puan=3-max&sayfa=",
            type="string",
        ),
        "hb_pages": Param(
            1, type="integer", description="HB Sayfa Sayısı- Her sayfa 36 ürün içerir"
        ),
        "ty_url": Param(
            "https://www.trendyol.com/sr?wc=103108%2C106084&sst=MOST_RATED",
            type="string",
        ),
        "ty_pages": Param(
            2, type="integer", description="TY Sayfa Sayısı - Her sayfa 16 ürün içerir"
        ),
    },
) as dag:
    start_pipeline = DummyOperator(task_id="start")

    # --- 1. SCRAPING (SSH - Host Makine) ---
    scrape_hb = SSHOperator(
        task_id="scrape_hepsiburada",
        ssh_conn_id="my_local_mac",
        command=f"""
            export DISPLAY=:0 && 
            cd {PROJECT_PATH} && 
            {VENV_PYTHON_PATH} {SCRIPT_NAME} \
            --site hb \
            --hb-pages {{{{ params.hb_pages }}}} \
            --hb-url "{{{{ params.hb_url }}}}"
        """,
        cmd_timeout=3600,
    )

    scrape_ty = SSHOperator(
        task_id="scrape_trendyol",
        ssh_conn_id="my_local_mac",
        command=f"""
            export DISPLAY=:0 && 
            cd {PROJECT_PATH} && 
            {VENV_PYTHON_PATH} {SCRIPT_NAME} \
            --site ty \
            --ty-pages {{{{ params.ty_pages }}}} \
            --ty-url "{{{{ params.ty_url }}}}"
        """,
        cmd_timeout=3600,
    )

    # --- 2. ETL (PythonOperator - Docker İçi) ---
    etl_process = PythonOperator(
        task_id="etl_feature_engineering", python_callable=run_project_etl
    )

    # --- 3. TRAINING (PythonOperator - Docker İçi) ---
    train_model = PythonOperator(
        task_id="train_model_process", python_callable=run_model_training
    )

    end_pipeline = DummyOperator(task_id="end")

    # --- AKIŞ ŞEMASI ---
    # Başla -> (HB ve TY Paralel) -> İkisi bitince ETL -> Sonra Training -> Bitiş
    (
        start_pipeline
        >> [scrape_hb, scrape_ty]
        >> etl_process
        >> train_model
        >> end_pipeline
    )
