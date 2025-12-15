import pandas as pd
import logging
from datetime import datetime
from sqlalchemy import text

# Proje İçi Modüller
from src.rfs.db.connector import get_db_engine
from src.rfs.features import extractors, parsers, engineering

# ----------------------------------------------------------------
# KONFİGÜRASYON VE SABİTLER
# ----------------------------------------------------------------

# Hepsiburada kolonlarını Trendyol şemasına eşleme
COL_MAPPING_HB_TO_TY = {
    "İşlemci": "İşlemci Modeli",
    "Maksimum İşlemci Hızı": "Maksimum İşlemci Hızı (GHz)",
    "Ram Tipi": "Ram (Sistem Belleği) Tipi",
    "Harddisk Kapasitesi": "Hard Disk Kapasitesi",
    "Max Ekran Çözünürlüğü": "Çözünürlük",
    "Ekran Özelliği": "Çözünürlük Standartı",
    "Ekran Panel Tipi": "Panel Tipi",
}

# Türkçe -> İngilizce Kolon Dönüşümü
COLS_EN = {
    "Başlık": "title",
    "Marka": "brand",
    "Kullanım Amacı": "intended_use",
    "Renk": "color",
    "Cihaz Ağırlığı": "weight",
    "İşlemci Tipi": "cpu_family",
    "İşlemci Modeli": "cpu_model",
    "İşlemci Nesli": "cpu_generation",
    "İşlemci Çekirdek Sayısı": "cpu_cores",
    "Maksimum İşlemci Hızı (GHz)": "cpu_max_ghz",
    "Ram (Sistem Belleği)": "ram_gb",
    "Ram (Sistem Belleği) Tipi": "ram_type",
    "Ekran Kartı": "gpu_model",
    "Ekran Kartı Tipi": "gpu_type",
    "Ekran Kartı Hafızası": "gpu_vram_gb",
    "Ekran Kartı Bellek Tipi": "gpu_vram_type",
    "SSD Kapasitesi": "ssd_gb",
    "Hard Disk Kapasitesi": "hdd_gb",
    "Ekran Boyutu": "screen_size_inch",
    "Çözünürlük": "resolution",
    "Çözünürlük Standartı": "display_standard",
    "Ekran Yenileme Hızı": "refresh_rate_hz",
    "Panel Tipi": "panel_type",
    "İşletim Sistemi": "operating_system",
    "Fiyat (TRY)": "price_try",
    "Link": "url",
    "Çekilme Zamanı": "scraped_at",
}

# Veritabanında görmek istediğimiz ideal sütun sırası
# (Transform ve Features tabloları için ortak temel)
FINAL_COLUMN_ORDER = [
    # Temel Bilgiler
    "title",
    "brand",
    "price_try",
    # Kullanım ve Fiziksel
    "intended_use",
    "color",
    "weight",
    "operating_system",
    # İşlemci (CPU)
    "cpu_family",
    "cpu_model",
    "cpu_generation",
    "cpu_cores",
    "cpu_max_ghz",
    # Bellek ve Depolama
    "ram_gb",
    "ram_type",
    "ssd_gb",
    "hdd_gb",
    # Ekran Kartı (GPU)
    "gpu_model",
    "gpu_type",
    "gpu_vram_gb",
    "gpu_vram_type",
    # Ekran (PPI buraya feature engineering adımında eklenecek)
    "screen_size_inch",
    "resolution",
    "display_standard",
    "refresh_rate_hz",
    "panel_type",
    # Meta Veriler
    "platform",
    "url",
    "scraped_at",
    "processed_at",
]

# ----------------------------------------------------------------
# PIPELINE SINIFI
# ----------------------------------------------------------------


class LaptopETLPipeline:
    def __init__(self):
        self.engine = get_db_engine()
        self.logger = logging.getLogger("etl.pipeline")
        # Log seviyesini ayarla
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

    def load_raw_data(self):
        """
        Raw tablolarından SADECE en son eklenen (güncel) veriyi çeker.
        Snapshot mantığı: DATE(created_at) == MAX(DATE(created_at))
        """
        try:
            # Hepsiburada için sorgu
            query_hb = """
            SELECT * FROM raw.hb 
            WHERE DATE(created_at) = (SELECT DATE(MAX(created_at)) FROM raw.hb)
            """

            # Trendyol için sorgu
            query_ty = """
            SELECT * FROM raw.ty 
            WHERE DATE(created_at) = (SELECT DATE(MAX(created_at)) FROM raw.ty)
            """

            df_hb = pd.read_sql(query_hb, self.engine)
            df_ty = pd.read_sql(query_ty, self.engine)

            self.logger.info(
                f"Loaded LATEST raw data only. HB: {len(df_hb)}, TY: {len(df_ty)}"
            )

            if df_hb.empty and df_ty.empty:
                self.logger.warning("Dikkat: Son işlem tarihinde hiç veri bulunamadı!")

            return df_hb, df_ty

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def merge_datasets(self, df_hb: pd.DataFrame, df_ty: pd.DataFrame) -> pd.DataFrame:
        """İki veri setini ortak şemada birleştirir."""
        # Platform etiketleri
        df_hb["platform"] = "hepsiburada"
        df_ty["platform"] = "trendyol"

        # Kolon eşleme (HB -> TY Standartı)
        df_hb = df_hb.rename(columns=COL_MAPPING_HB_TO_TY)

        # Ortak kolonları al
        common_cols = list(set(df_hb.columns) & set(df_ty.columns))
        df_merged = pd.concat(
            [df_ty[common_cols], df_hb[common_cols]], ignore_index=True
        )

        return df_merged

    def clean_and_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Veri temizleme, dönüştürme ve zenginleştirme (Silver Layer)."""

        # --- GENEL TEMİZLİK ---
        # Tüm object (string) tipli kolonlarda 'yok', 'belirtilmemiş' gibi ifadeleri temizle
        object_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in object_cols:
            df[col] = df[col].apply(parsers.clean_garbage_text)

        # 1. Başlıktan Veri Doldurma (Imputation - Deterministic)
        # -------------------------------------------------------
        df = extractors.fill_missing_from_title(
            df, "Başlık", "Ram (Sistem Belleği)", extractors.extract_ram_from_title
        )
        df = extractors.fill_missing_from_title(
            df, "Başlık", "SSD Kapasitesi", extractors.extract_ssd_from_title
        )
        df = extractors.fill_missing_from_title(
            df,
            "Başlık",
            "Ekran Yenileme Hızı",
            extractors.extract_refresh_rate_from_title,
        )
        df = extractors.fill_missing_from_title(
            df,
            "Başlık",
            "Çözünürlük Standartı",
            extractors.extract_resolution_from_title,
        )

        # 2. Kolon İsimlendirme (TR -> EN)
        # --------------------------------
        df = df.rename(columns=COLS_EN)

        # 3. Apple Filtreleme
        # -------------------
        mask_apple = df["brand"].astype(str).str.contains(
            "apple", case=False, na=False
        ) | df["title"].astype(str).str.contains("macbook", case=False, na=False)
        df = df[~mask_apple].copy()

        # 4. Duplicate Temizliği (Title + Price bazlı)
        # --------------------------------------------
        df = df.drop_duplicates(subset=["title", "price_try"], keep="last")

        # 5. Column Parsing (Standartlaştırma)
        # ------------------------------------
        # Numeric Fields
        df["price_try"] = df["price_try"].apply(parsers.parse_price)
        df["cpu_generation"] = df["cpu_generation"].apply(
            lambda x: parsers.parse_numeric(x, 1, 15)
        )
        df["cpu_cores"] = df["cpu_cores"].apply(
            lambda x: parsers.parse_numeric(x, 1, 15)
        )
        df["cpu_max_ghz"] = df["cpu_max_ghz"].apply(
            lambda x: parsers.parse_numeric(x, 1, 6)
        )
        df["ram_gb"] = df["ram_gb"].apply(lambda x: parsers.parse_numeric(x, 4, 128))
        df["ssd_gb"] = df["ssd_gb"].apply(lambda x: parsers.parse_numeric(x, 120, 8192))
        df["hdd_gb"] = df["hdd_gb"].apply(lambda x: parsers.parse_numeric(x, 120, 8192))
        df["gpu_vram_gb"] = df["gpu_vram_gb"].apply(
            lambda x: parsers.parse_gpu_vram(x, 0, 32)
        )
        df["screen_size_inch"] = df["screen_size_inch"].apply(
            lambda x: parsers.parse_numeric(x, 10, 20)
        )
        df["refresh_rate_hz"] = df["refresh_rate_hz"].apply(
            lambda x: parsers.parse_numeric(x, 30, 360)
        )

        # Categorical Fields
        df["brand"] = df["brand"].apply(parsers.parse_brand)
        df["operating_system"] = df["operating_system"].apply(parsers.parse_os)
        df["cpu_family"] = df["cpu_family"].apply(parsers.parse_cpu_family)
        # gpu model ?
        df["gpu_type"] = df["gpu_type"].apply(parsers.parse_gpu_type)
        # gpu vram type ?
        df["resolution"] = df["resolution"].apply(parsers.parse_resolution)
        # display standard ?
        df["panel_type"] = df["panel_type"].apply(parsers.parse_panel_type)

        # 6. Null Fiyatları Temizle
        df = df.dropna(subset=["price_try"])

        return df

    def feature_engineering_step(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature Engineering (Gold Layer). PPI ve Gruplama."""
        self.logger.info("Applying Feature Engineering (PPI, Grouping)...")

        # engineering.py modülünü kullan
        df_engineered = engineering.apply_feature_engineering(df)
        return df_engineered

    def save_to_db(self, df: pd.DataFrame, schema: str, table: str):
        """Genel veritabanı kayıt fonksiyonu. Sütun sırasını düzenler ve kaydeder."""
        try:
            # Şemanın varlığından emin ol
            with self.engine.connect() as conn:
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
                # conn.commit()

            df["processed_at"] = datetime.now()

            # --- Sütun Sıralaması (Reordering) ---
            # PPI gibi yeni eklenen kolonları da kapsayacak şekilde dinamik liste oluştur
            target_order = FINAL_COLUMN_ORDER.copy()

            # Eğer 'ppi' hesaplandıysa ve df'de varsa, 'screen_size_inch' yanına ekleyelim
            if "ppi" in df.columns and "ppi" not in target_order:
                idx = target_order.index("screen_size_inch")
                target_order.insert(idx + 1, "ppi")

            # Sadece df'de gerçekten var olan kolonları seç (Hata önleyici)
            cols_to_write = [c for c in target_order if c in df.columns]

            # Listede olmayan ama df'de kalan kolonlar varsa (varsa sona ekle)
            remaining = [c for c in df.columns if c not in cols_to_write]

            df = df[cols_to_write + remaining]

            # DB'ye Yaz (Replace mantığı ile)
            df.to_sql(
                name=table,
                con=self.engine,
                schema=schema,
                if_exists="replace",  # Tabloyu sil ve yeniden oluştur (Snapshot)
                index=False,
            )
            self.logger.info(f"Saved to {schema}.{table}: {len(df)} rows")

        except Exception as e:
            self.logger.error(f"Error saving to {schema}.{table}: {e}")
            raise

    def run(self):
        """Pipeline Akışı: Extract -> Transform -> Feature Eng -> Load"""
        self.logger.info("Starting ETL Pipeline...")

        # 1. Extract (En Güncel Veri)
        df_hb, df_ty = self.load_raw_data()

        # 2. Merge
        df_merged = self.merge_datasets(df_hb, df_ty)

        # 3. Transform (Silver Layer - Temiz Veri)
        df_clean = self.clean_and_transform(df_merged)
        self.save_to_db(df_clean, "transform", "laptops")

        # 4. Feature Engineering (Gold Layer - Model Verisi)
        df_final = self.feature_engineering_step(df_clean)
        self.save_to_db(df_final, "features", "laptops_final")

        self.logger.info("ETL Pipeline completed successfully.")
