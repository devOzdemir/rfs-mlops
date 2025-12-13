import pandas as pd
import logging
from datetime import datetime

from src.rfs.db.connector import get_db_engine
from src.rfs.features import extractors, parsers

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
    # Ekran
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
        # Log seviyesini ayarla (konsolda görmek için)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

    def load_raw_data(self):
        """Raw tablolarından (hb ve ty) veriyi çeker."""
        try:
            # Pandas read_sql kullanarak şemaları belirtiyoruz
            df_hb = pd.read_sql("SELECT * FROM raw.hb", self.engine)
            df_ty = pd.read_sql("SELECT * FROM raw.ty", self.engine)

            self.logger.info(f"Loaded raw data. HB: {len(df_hb)}, TY: {len(df_ty)}")
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
        """Veri temizleme, dönüştürme ve zenginleştirme."""

        # 1. Başlıktan Veri Doldurma (Imputation)
        # ---------------------------------------
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

        # 2. Kolon İsimlendirme (TR -> EN)
        # --------------------------------
        df = df.rename(columns=COLS_EN)

        # 3. Apple Filtreleme (Veri setini kirletmemesi için)
        # -------------------
        mask_apple = df["brand"].astype(str).str.contains(
            "apple", case=False, na=False
        ) | df["title"].astype(str).str.contains("macbook", case=False, na=False)
        df = df[~mask_apple].copy()

        # 4. Duplicate Temizliği (Title + Price bazlı)
        # ----------------------
        df = df.drop_duplicates(subset=["title", "price_try"], keep="last")

        # 5. Column Parsing (Standartlaştırma)
        # ------------------------------------
        # Numeric Fields
        df["price_try"] = df["price_try"].apply(parsers.parse_price)
        df["ram_gb"] = df["ram_gb"].apply(lambda x: parsers.parse_numeric(x, 4, 128))
        df["ssd_gb"] = df["ssd_gb"].apply(lambda x: parsers.parse_numeric(x, 120, 8192))
        df["screen_size_inch"] = df["screen_size_inch"].apply(
            lambda x: parsers.parse_numeric(x, 10, 20)
        )
        df["cpu_generation"] = df["cpu_generation"].apply(
            lambda x: parsers.parse_numeric(x, 1, 15)
        )

        # Categorical Fields
        df["brand"] = df["brand"].apply(parsers.parse_brand)
        df["operating_system"] = df["operating_system"].apply(parsers.parse_os)
        df["gpu_type"] = df["gpu_type"].apply(parsers.parse_gpu_type)
        df["panel_type"] = df["panel_type"].apply(parsers.parse_panel_type)
        df["cpu_family"] = df["cpu_family"].apply(parsers.parse_cpu_family)
        df["resolution"] = df["resolution"].apply(parsers.parse_resolution)

        # 6. Null Fiyatları Temizle (Model için kritik)
        df = df.dropna(subset=["price_try"])

        # 7. Sütun Sıralamasını Düzenle (Reordering)
        # -------------------------------------------
        # Sadece veri setinde gerçekten var olan kolonları seç (Hata almamak için)
        final_cols = [c for c in FINAL_COLUMN_ORDER if c in df.columns]

        # Varsa listede unuttuğumuz ama df'de olan diğer kolonları da sona ekle
        remaining_cols = [c for c in df.columns if c not in final_cols]

        df = df[final_cols + remaining_cols]

        return df

    def save_transformed_data(self, df: pd.DataFrame):
        """İşlenmiş veriyi transform şemasına yazar."""
        try:
            # İşlenme tarihini ekle
            df["processed_at"] = datetime.now()

            # DB'ye yaz (replace: her seferinde tabloyu sıfırla ve yeniden oluştur)
            # Bu sayede sütun sırası da veritabanında düzelir.
            df.to_sql(
                name="laptops",
                con=self.engine,
                schema="transform",
                if_exists="replace",
                index=False,
            )
            self.logger.info(
                f"Saved transformed data: {len(df)} rows to transform.laptops"
            )
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            raise

    def run(self):
        """Pipeline'ı çalıştırır."""
        self.logger.info("Starting ETL Pipeline...")

        # Extract
        df_hb, df_ty = self.load_raw_data()

        # Merge
        df_merged = self.merge_datasets(df_hb, df_ty)

        # Transform (Cleaning + Reordering)
        df_clean = self.clean_and_transform(df_merged)

        # Load
        self.save_transformed_data(df_clean)

        self.logger.info("ETL Pipeline completed successfully.")
