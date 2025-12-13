# Database connection logic (SQLAlchemy)
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
        self.user = os.getenv("POSTGRES_USER", "rfs_user")
        self.password = os.getenv("POSTGRES_PASSWORD", "rfs_password")
        self.host = os.getenv("POSTGRES_HOST", "localhost")
        self.port = os.getenv("POSTGRES_PORT", "5432")
        self.dbname = os.getenv("POSTGRES_DB", "rfs_db")

        # SQLAlchemy Connection String
        self.db_url = f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"
        self.engine = create_engine(self.db_url)
        logging.info("Veritabanı motoru başlatıldı.")

    def get_engine(self):
        return self.engine


# Singleton kullanımı için helper fonksiyon
def get_db_engine():
    return DatabaseConnector().get_engine()
