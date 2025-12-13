import os
import logging
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from sqlalchemy.types import String, Text, Integer, DateTime

# DB Bağlantısı
from src.rfs.db.connector import get_db_engine


class BaseScraper(ABC):
    def __init__(self, platform_name: str, output_dir: str = "data"):
        self.platform_name = platform_name
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M")

        # --- Dosya Yolları (Eski CSV Mantığı) ---
        self.base_output_dir = Path(output_dir)
        self.link_dir = self.base_output_dir / "links"
        self.raw_dir = self.base_output_dir / "raw"
        self.logs_dir = self.base_output_dir / "logs"
        self._create_directories()

        # --- Loglama ---
        self.logger = self._setup_logger()
        self.driver = None

        # --- DB Engine ---
        try:
            self.engine = get_db_engine()
        except Exception as e:
            self.logger.error(f"DB Bağlantısı başarısız: {e}")
            self.engine = None

    def _create_directories(self):
        """Gerekli klasörleri oluşturur."""
        for directory in [self.link_dir, self.raw_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        """Her platform için izole edilmiş logger oluşturur."""
        logger = logging.getLogger(f"scraper.{self.platform_name}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            log_file = self.logs_dir / f"{self.platform_name}_{self.timestamp}.log"
            file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
        return logger

    def start_driver(self, headless: bool = False):
        """Selenium WebDriver'ı başlatır."""
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        # chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        self.logger.info("WebDriver başlatılıyor...")
        self.driver = webdriver.Chrome(options=chrome_options)

    def close_driver(self):
        if self.driver:
            self.logger.info("WebDriver kapatılıyor...")
            self.driver.quit()
            self.driver = None

    def save_data(self, df: pd.DataFrame, file_prefix: str, sub_folder: str = "raw"):
        """
        Veriyi kaydeder.
        CSV kaydı yorum satırına alındı (debugging için açılabilir).
        DB kaydı aktif edildi.
        """
        if df.empty:
            self.logger.warning("Kaydedilecek veri bulunamadı.")
            return

        # --- 1. YÖNTEM: ESKİ CSV KAYDI (DEBUG İÇİN TUTULUYOR) ---
        # target_dir = self.base_output_dir / sub_folder
        # filename = f"{file_prefix}_{self.platform_name}_{self.timestamp}.csv"
        # file_path = target_dir / filename
        # df.to_csv(file_path, index=False)
        # self.logger.info(f"[CSV] Veri dosyaya yazıldı: {file_path}")

        # --- 2. YÖNTEM: VERİTABANI KAYDI ---
        if self.engine:
            try:
                # --- Şema ve Tablo Belirleme ---
                target_schema = "public"

                # Tablo adı her zaman platform kısaltması (hb veya ty)
                target_table = self.platform_name.lower()

                if sub_folder == "links":
                    target_schema = "links"
                elif sub_folder == "raw":
                    target_schema = "raw"

                # Timestamp ekle
                if "created_at" not in df.columns:
                    df["created_at"] = datetime.now()

                # --- Link Kontrolü (Deduplication) ---
                # Linkler tablosunda aynı linkten varsa tekrar eklememek için basit bir önlem (Opsiyonel ama önerilir)
                # Şimdilik sadece append yapıyoruz, ileride buraya "upsert" mantığı eklenebilir.

                # DB'ye Yaz
                df.to_sql(
                    name=target_table,
                    con=self.engine,
                    schema=target_schema,
                    if_exists="append",
                    index=False,
                    dtype={"Link": Text, "Name": Text},
                )

                self.logger.info(
                    f"[DB] Veri {target_schema}.{target_table} tablosuna basıldı. ({len(df)} satır)"
                )

            except Exception as e:
                self.logger.error(
                    f"[DB Error] {target_schema}.{target_table} yazma hatası: {e}"
                )
        else:
            self.logger.error("DB Engine yok, kayıt yapılamadı!")

    # ... (Abstract metodlar aynı kalıyor) ...
    @abstractmethod
    def scrape_links(self, base_url: str, total_pages: int) -> pd.DataFrame:
        pass

    @abstractmethod
    def scrape_details(self, links_df: pd.DataFrame) -> pd.DataFrame:
        pass
