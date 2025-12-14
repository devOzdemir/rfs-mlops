-- init_db.sql

-- Şemalar (Schemas)
CREATE SCHEMA IF NOT EXISTS links;
CREATE SCHEMA IF NOT EXISTS raw;      -- Bronze Layer (Raw Data)
CREATE SCHEMA IF NOT EXISTS transform; -- Silver Layer (Clean Data)
CREATE SCHEMA IF NOT EXISTS features;  -- Gold Layer (Model Ready Data)
CREATE SCHEMA IF NOT EXISTS logs;

-- İzinler (Permissions)
-- rfs_user kullanıcısına tüm şemalarda tam yetki veriyoruz
GRANT ALL PRIVILEGES ON SCHEMA links TO rfs_user;
GRANT ALL PRIVILEGES ON SCHEMA raw TO rfs_user;
GRANT ALL PRIVILEGES ON SCHEMA transform TO rfs_user;
GRANT ALL PRIVILEGES ON SCHEMA features TO rfs_user;
GRANT ALL PRIVILEGES ON SCHEMA logs TO rfs_user;

-- Gelecekte oluşturulacak tablolar için de yetki veriyoruz
ALTER DEFAULT PRIVILEGES IN SCHEMA links GRANT ALL ON TABLES TO rfs_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA raw GRANT ALL ON TABLES TO rfs_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA transform GRANT ALL ON TABLES TO rfs_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA features GRANT ALL ON TABLES TO rfs_user;