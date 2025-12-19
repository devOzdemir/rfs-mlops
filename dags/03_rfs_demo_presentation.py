from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.models.param import Param
from datetime import datetime, timedelta
import sys
import os
import pandas as pd
import logging

# Docker path ayarı
sys.path.append("/opt/airflow")

# Mevcut Transformation Mantığını kullanmak için import
# (Feature engineering ve cleaning kodlarını tekrar yazmıyoruz, çağırıyoruz)
from src.rfs.features.pipeline import LaptopETLPipeline

# --- AYARLAR ---
# Airflow Docker içindedir, bu yüzden path'i Docker'ın gördüğü şekilde veriyoruz.
# Scraper (Host) "dags/demo_data"ya yazdı, Airflow da "/opt/airflow/dags/demo_data"dan okuyacak.
DEMO_DATA_DIR = "/opt/airflow/dags/demo_data"

# Host makine ayarları (.env'den)
VENV_PYTHON_PATH = os.getenv("VENV_PYTHON_PATH")
PROJECT_PATH = os.getenv("PROJECT_PATH")
SSH_CONN_ID = os.getenv("MAC_SSH_NAME", "my_local_mac")
SCRIPT_NAME = "run_scraping.py"


def run_demo_etl_no_db():
    """
    DB kullanmadan, sadece CSV dosyaları üzerinden ETL işlemini simüle eder.
    Pipeline sınıfındaki temizleme fonksiyonlarını kullanır ama I/O işlemini dosya sistemiyle yapar.
    """
    print("🚀 Sunum Modu: Dosya Tabanlı ETL Başlatılıyor...")

    # 1. EXTRACT: Scraper'ın ürettiği CSV'leri oku
    try:
        raw_hb_path = f"{DEMO_DATA_DIR}/raw/hb_raw.csv"
        raw_ty_path = f"{DEMO_DATA_DIR}/raw/ty_raw.csv"

        print(f"📂 Okunuyor: {raw_hb_path}")
        df_hb = pd.read_csv(raw_hb_path)

        print(f"📂 Okunuyor: {raw_ty_path}")
        df_ty = pd.read_csv(raw_ty_path)

        print(f"📊 Ham Veri Yüklendi -> HB: {len(df_hb)}, TY: {len(df_ty)}")
    except FileNotFoundError as e:
        print(f"❌ HATA: Demo dosyaları bulunamadı. Önce Scraper çalışmalı! Hata: {e}")
        raise e

    # Pipeline Sınıfını Başlat (Sadece metodlarına erişmek için, DB bağlanmasa da olur)
    pipeline = LaptopETLPipeline()

    # 2. MERGE
    print("🔄 Veriler Birleştiriliyor...")
    df_merged = pipeline.merge_datasets(df_hb, df_ty)

    # 3. TRANSFORM (Temizleme)
    print("🧹 Veri Temizleniyor (Silver Layer)...")
    # Clean fonksiyonu DB gerektirmez, Pandas üzerinde çalışır
    df_clean = pipeline.clean_and_transform(df_merged)

    # Ara Kayıt (Silver)
    os.makedirs(f"{DEMO_DATA_DIR}/processed", exist_ok=True)
    silver_path = f"{DEMO_DATA_DIR}/processed/demo_silver_clean.csv"
    df_clean.to_csv(silver_path, index=False)
    print(f"✅ Temiz veri kaydedildi: {silver_path}")

    # 4. FEATURE ENGINEERING (Gold Layer)
    print("✨ Özellik Mühendisliği (Gold Layer)...")
    df_final = pipeline.feature_engineering_step(df_clean)

    # Son Kayıt (Gold)
    os.makedirs(f"{DEMO_DATA_DIR}/final", exist_ok=True)
    gold_path = f"{DEMO_DATA_DIR}/final/demo_gold_features.csv"
    df_final.to_csv(gold_path, index=False)

    print(f"🎉 Demo ETL Tamamlandı!")
    print(f"📂 Dosyaları şurada gösterebilirsiniz: rfs-mlops/dags/demo_data/")


default_args = {
    "owner": "presentation_mode",
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="03_rfs_demo_presentation",
    default_args=default_args,
    description="SUNUM MODU: DB Yok! Scraping -> CSV -> ETL -> CSV",
    schedule_interval=timedelta(hours=2),
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["demo", "no-db", "presentation"],
    params={
        "hb_pages": Param(1, type="integer", description="HB Sayfa Sayısı"),
        "ty_pages": Param(1, type="integer", description="TY Sayfa Sayısı"),
    },
) as dag:
    start = DummyOperator(task_id="start_demo")

    # 1. SCRAPER (Host Makine)
    # Kritik Nokta: "DEMO_MODE=true" veriyoruz.
    # BaseScraper bunu görünce DB yerine dags/demo_data/raw klasörüne CSV yazacak.

    scrape_hb = SSHOperator(
        task_id="scrape_hepsiburada_csv",
        ssh_conn_id=SSH_CONN_ID,
        command=f"""
            export DEMO_MODE=true && \
            export DISPLAY=:0 && \
            cd {PROJECT_PATH} && \
            {VENV_PYTHON_PATH} {SCRIPT_NAME} \
            --site hb \
            --hb-pages {{{{ params.hb_pages }}}}
        """,
        cmd_timeout=600,
    )

    scrape_ty = SSHOperator(
        task_id="scrape_trendyol_csv",
        ssh_conn_id=SSH_CONN_ID,
        command=f"""
            export DEMO_MODE=true && \
            export DISPLAY=:0 && \
            cd {PROJECT_PATH} && \
            {VENV_PYTHON_PATH} {SCRIPT_NAME} \
            --site ty \
            --ty-pages {{{{ params.ty_pages }}}}
        """,
        cmd_timeout=600,
    )

    # 2. ETL (Dosya Tabanlı)
    etl_files = PythonOperator(
        task_id="etl_process_files",
        python_callable=run_demo_etl_no_db,
    )

    end = DummyOperator(task_id="end_demo")

    # Akış
    start >> [scrape_hb, scrape_ty] >> etl_files >> end
