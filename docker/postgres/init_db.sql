-- init_db.sql

-- İstenilen Şemaların (Schema) Oluşturulması
CREATE SCHEMA IF NOT EXISTS links;
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS logs;
CREATE SCHEMA IF NOT EXISTS transform;

-- İzinlerin Ayarlanması (Gerekirse)
GRANT ALL PRIVILEGES ON SCHEMA links TO rfs_user;
GRANT ALL PRIVILEGES ON SCHEMA raw TO rfs_user;
GRANT ALL PRIVILEGES ON SCHEMA logs TO rfs_user;
GRANT ALL PRIVILEGES ON SCHEMA transform TO rfs_user;