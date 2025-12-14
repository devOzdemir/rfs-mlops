# src/rfs/db/connector.py
import os
import logging
from sqlalchemy import create_engine
from dotenv import load_dotenv

# .env yükle
load_dotenv()


class DatabaseConnector:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnector, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # GÜNCELLEME: Değişken isimlerini yeni .env dosyana göre ayarladık.
        # Artık admin (POSTGRES_...) değil, uygulama kullanıcısı (RFS_...) bilgilerini çekiyoruz.

        self.user = os.getenv("RFS_DB_USER")
        self.password = os.getenv("RFS_DB_PASSWORD")
        self.dbname = os.getenv("RFS_DB_NAME")

        # Eğer bilgisayarından (VS Code) çalıştırıyorsan 'localhost',
        # Docker içinden çalıştırırsan 'postgres' servisini görecek şekilde ayarlayalım:
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = os.getenv("DB_PORT", "5432")

        # Eksik bilgi kontrolü (Hata ayıklamayı kolaylaştırır)
        if not all([self.user, self.password, self.dbname]):
            logging.error(
                "❌ Veritabanı bilgileri .env dosyasından okunamadı! RFS_DB_... değişkenlerini kontrol et."
            )
            raise ValueError("Eksik Environment Değişkenleri")

        # SQLAlchemy Connection String
        self.db_url = f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"

        try:
            self.engine = create_engine(self.db_url)
            logging.info(
                f"Veritabanı motoru başlatıldı. Hedef: {self.host}:{self.port}/{self.dbname}"
            )
        except Exception as e:
            logging.error(f"Veritabanı motoru oluşturulurken hata: {e}")
            raise e

    def get_engine(self):
        return self.engine


# Singleton kullanımı için helper fonksiyon
def get_db_engine():
    return DatabaseConnector().get_engine()


# --- TEST KISMI (Dosyayı doğrudan çalıştırdığında bağlantıyı dener) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            print("\n✅ BAŞARILI: Veritabanına RFS kullanıcısı ile bağlandın!")
    except Exception as e:
        print(f"\n❌ HATA: Bağlantı başarısız. Detay: {e}")
