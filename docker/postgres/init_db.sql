-- BU SCRIPT "POSTGRES_USER" (admin_chef) YETKİSİYLE ÇALIŞIR

-- ==========================================
-- 1. KULLANICILARI OLUŞTURMA
-- ==========================================

-- RFS Kullanıcısı (Eğer yoksa oluştur)
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'rfs_user') THEN
    CREATE ROLE rfs_user WITH LOGIN PASSWORD 'rfs_pass';
  END IF;
END
$$;

-- Airflow Kullanıcısı
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'airflow_user') THEN
    CREATE ROLE airflow_user WITH LOGIN PASSWORD 'airflow_pass';
  END IF;
END
$$;

-- MLflow Kullanıcısı
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'mlflow_user') THEN
    CREATE ROLE mlflow_user WITH LOGIN PASSWORD 'mlflow_pass';
  END IF;
END
$$;


-- ==========================================
-- 2. VERİTABANLARINI OLUŞTURMA
-- ==========================================
-- Not: rfs_db zaten docker-compose'daki POSTGRES_DB ile otomatik oluşur.
-- Biz diğerlerini ekleyeceğiz.

-- Airflow Veritabanı (Sahibi: airflow_user)
SELECT 'CREATE DATABASE airflow_db OWNER airflow_user'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow_db')\gexec

-- MLflow Veritabanı (Sahibi: mlflow_user)
SELECT 'CREATE DATABASE mlflow_db OWNER mlflow_user'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow_db')\gexec

-- RFS Veritabanının sahibini rfs_user yapalım (Admin yaratmıştı)
ALTER DATABASE rfs_db OWNER TO rfs_user;


-- ==========================================
-- 3. RFS PROJE ŞEMALARI (rfs_db İÇİN)
-- ==========================================
-- Şu an rfs_db içindeyiz (Varsayılan DB olduğu için).
-- Buraya senin proje tabloların gelecek.

-- Şemalar
CREATE SCHEMA IF NOT EXISTS links AUTHORIZATION rfs_user;
CREATE SCHEMA IF NOT EXISTS raw AUTHORIZATION rfs_user;
CREATE SCHEMA IF NOT EXISTS transform AUTHORIZATION rfs_user;
CREATE SCHEMA IF NOT EXISTS features AUTHORIZATION rfs_user;
CREATE SCHEMA IF NOT EXISTS logs AUTHORIZATION rfs_user;

-- MLflow ve Airflow kendi veritabanlarını kullanacağı için
-- buradaki "public" şemasına karışmalarına gerek yok.
-- Sadece rfs_user'a yetki veriyoruz.

GRANT ALL PRIVILEGES ON SCHEMA links TO rfs_user;
GRANT ALL PRIVILEGES ON SCHEMA raw TO rfs_user;
GRANT ALL PRIVILEGES ON SCHEMA transform TO rfs_user;
GRANT ALL PRIVILEGES ON SCHEMA features TO rfs_user;
GRANT ALL PRIVILEGES ON SCHEMA logs TO rfs_user;
GRANT ALL PRIVILEGES ON SCHEMA public TO rfs_user;

-- Gelecek tablolar için yetkiler
ALTER DEFAULT PRIVILEGES IN SCHEMA links GRANT ALL ON TABLES TO rfs_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA raw GRANT ALL ON TABLES TO rfs_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA transform GRANT ALL ON TABLES TO rfs_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA features GRANT ALL ON TABLES TO rfs_user;