-- ŞEMALARIN OLUŞTURULMASI
CREATE SCHEMA IF NOT EXISTS links;      -- Linklerin tutulduğu yer
CREATE SCHEMA IF NOT EXISTS raw;        -- Ham veriler (Bronze)
CREATE SCHEMA IF NOT EXISTS transform;  -- Temiz veriler (Silver)
CREATE SCHEMA IF NOT EXISTS features;   -- Modele girecek veriler (Gold) - YENİ
CREATE SCHEMA IF NOT EXISTS logs;       -- Log kayıtları

-- İZİNLER (rfs_user için)
-- 1. Şemalar üzerindeki izinler
GRANT ALL PRIVILEGES ON SCHEMA links TO rfs_user;
GRANT ALL PRIVILEGES ON SCHEMA raw TO rfs_user;
GRANT ALL PRIVILEGES ON SCHEMA transform TO rfs_user;
GRANT ALL PRIVILEGES ON SCHEMA features TO rfs_user;
GRANT ALL PRIVILEGES ON SCHEMA logs TO rfs_user;
-- MLflow varsayılan olarak public şemasını kullanır, oraya da yetki verelim
GRANT ALL PRIVILEGES ON SCHEMA public TO rfs_user;

-- 2. Tablolar üzerindeki izinler (Gelecekteki tablolar için)
ALTER DEFAULT PRIVILEGES IN SCHEMA links GRANT ALL ON TABLES TO rfs_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA raw GRANT ALL ON TABLES TO rfs_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA transform GRANT ALL ON TABLES TO rfs_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA features GRANT ALL ON TABLES TO rfs_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA logs GRANT ALL ON TABLES TO rfs_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO rfs_user;