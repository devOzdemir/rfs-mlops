# Abstract Base Class for Scrapers
import os
import logging
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


class BaseScraper(ABC):
    """
    Tüm web scraper modülleri için temel soyut sınıf.
    Driver yönetimi, loglama ve veri kaydetme işlemlerini standartlaştırır.
    """

    def __init__(self, platform_name: str, output_dir: str = "data"):
        self.platform_name = platform_name
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M")

        # Dizin Ayarları
        self.base_output_dir = Path(output_dir)
        self.link_dir = self.base_output_dir / "links"
        self.raw_dir = self.base_output_dir / "raw"
        self.logs_dir = self.base_output_dir / "logs"

        self._create_directories()
        self.logger = self._setup_logger()
        self.driver = None

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

            # Dosya Handler
            log_file = self.logs_dir / f"{self.platform_name}_{self.timestamp}.log"
            file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            # Konsol Handler
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        return logger

    def start_driver(self, headless: bool = False):
        """Selenium WebDriver'ı başlatır."""
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")

        # Anti-bot tespiti için temel argümanlar
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")

        self.logger.info("WebDriver başlatılıyor...")
        self.driver = webdriver.Chrome(options=chrome_options)

    def close_driver(self):
        """Driver'ı güvenli şekilde kapatır."""
        if self.driver:
            self.logger.info("WebDriver kapatılıyor...")
            self.driver.quit()
            self.driver = None

    def save_data(self, df: pd.DataFrame, file_prefix: str, sub_folder: str = "raw"):
        """Veriyi CSV olarak kaydeder."""
        if df.empty:
            self.logger.warning("Kaydedilecek veri bulunamadı (DataFrame boş).")
            return

        target_dir = self.base_output_dir / sub_folder
        filename = f"{file_prefix}_{self.platform_name}_{self.timestamp}.csv"
        file_path = target_dir / filename

        df.to_csv(file_path, index=False)
        self.logger.info(f"Veri kaydedildi: {file_path}")

    @abstractmethod
    def scrape_links(self, base_url: str, total_pages: int) -> pd.DataFrame:
        """Ürün linklerini toplama mantığı (Override edilmeli)."""
        pass

    @abstractmethod
    def scrape_details(self, links_df: pd.DataFrame) -> pd.DataFrame:
        """Detay sayfalarını gezme mantığı (Override edilmeli)."""
        pass
