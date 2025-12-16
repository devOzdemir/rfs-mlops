# 🚀 RFS - Rekabetçi Fiyatlandırma Sistemi (Laptop Price Prediction)

**RFS**, e-ticaret sitelerinden (Hepsiburada, Trendyol) anlık veri toplayan, makine öğrenmesi modelleriyle fiyat tahmini yapan ve sonuçları canlı bir web arayüzünde sunan uçtan uca (End-to-End) bir **MLOps** projesidir.

Proje; Veri Mühendisliği, Model Eğitimi (Experiment Tracking) ve Model Sunumu (Serving) süreçlerinin tamamını **Docker** üzerinde mikroservis mimarisiyle yönetir.

---

## 🏗️ Mimari ve Teknolojiler

Proje **6 ana bileşenden** oluşur:

1.  **Orkestrasyon (Apache Airflow):** Veri kazıma (Scraping) ve model eğitim süreçlerini zamanlar ve yönetir.
2.  **Experiment Tracking (MLflow):** Eğitilen modellerin parametrelerini, başarı metriklerini (RMSE, MAE) ve versiyonlarını saklar. En iyi modeli otomatik olarak **`@champion`** olarak etiketler.
3.  **Veri İşleme (Scikit-Learn & Pandas):** Ham veriyi temizler, eksik verileri doldurur (Imputation) ve özellik mühendisliği (Feature Engineering) yapar.
4.  **Model API (FastAPI):** `@champion` etiketli modeli canlıya alır. Gelen istekleri doğrular (Pydantic) ve fiyat tahmini döner.
5.  **Kullanıcı Arayüzü (Flask & Bootstrap):** Kullanıcıların kolayca tahmin alabileceği, dinamik formlara sahip web arayüzü.
6.  **Veri Tabanı & Depolama:** PostgreSQL (Airflow/MLflow metadata için) ve MinIO (Model artifactleri için).

![Tech Stack](https://skillicons.dev/icons?i=python,docker,fastapi,flask,scikitlearn,postgres,bootstrap)

---

## ✨ Temel Özellikler

* **🔄 Tam Otomasyon:** Tek bir Airflow DAG'ı ile veri çekme -> temizleme -> eğitim -> dağıtım süreci otomatik işler.
* **🧠 Akıllı Model Seçimi:** Sistem birden fazla algoritmayı (XGBoost, RandomForest, Ridge vb.) yarıştırır ve en düşük hata oranına sahip olanı "Production"a alır.
* **🎛️ Dynamic Dropdowns:** API, eğitim verisindeki marka ve modelleri (Örn: "RTX 4060", "Asus") otomatik öğrenir. UI, kod değişikliği gerekmeden kendini günceller.
* **⚡ Feature Engineering:** Kullanıcıdan "Çözünürlük" ve "İnç" bilgisini alır, arka planda **PPI (Pixel Per Inch)** hesaplayarak modele verir.
* **📜 Client-Side History:** Kullanıcının yaptığı sorgular tarayıcı hafızasında (LocalStorage) tutulur, veri tabanı maliyeti yaratmaz.

---

## 📂 Proje Yapısı

```bash
.
├── api/                   # FastAPI Kodları (Serving)
│   ├── main.py            # API Endpointleri
│   └── schemas.py         # Pydantic Veri Doğrulama
├── configs/               # Model ve Veri Konfigürasyonları (YAML)
├── dags/                  # Airflow İş Akışları (ETL & Training)
├── docker/                # Dockerfile ve Altyapı Dosyaları
│   ├── airflow/
│   ├── api/
│   └── ui/
├── src/                   # Çekirdek ML Kodları (Training, Preprocessing)
├── ui/                    # Flask Web Arayüzü
│   ├── app.py
│   └── templates/
└── docker-compose.yaml    # Tüm servislerin orkestrasyonu
```

---

## 🚀 Kurulum ve Çalıştırma

Projeyi yerel makinenizde çalıştırmak için **Docker** ve **Docker Compose** yüklü olmalıdır.

### 1. Hazırlık: Ortam Değişkenleri (.env)

Sistemin hibrit yapısının (Docker içinden Host makineye SSH ile bağlanıp tarayıcı açması) çalışabilmesi için kimlik bilgilerinizi tanımlamanız gerekir.

1.  Proje ana dizinindeki `.env.example` dosyasının adını `.env` olarak değiştirin.
2.  Dosyayı açın ve aşağıdaki alanları **kendi bilgisayarınızın** kullanıcı adı ve şifresiyle doldurun:

```ini
# Host makineye (Kendi bilgisayarınıza) bağlanmak için
MAC_SSH_USER=bilgisayarinizin_kullanici_adi
MAC_SSH_PASSWORD=bilgisayarinizin_sifresi
MAC_SSH_NAME=my_local_mac # airflowdaki ad degistirmeye gerek yok 
VENV_PYTHON_PATH=`proje dosya yolu`/rfs-mlops/.venv/bin/python
PROJECT_PATH=`proje dosya yolu`/rfs-mlops
MAC_EXEC_PATH=/usr/local/bin:/opt/homebrew/bin #Docker komutlari icin path degiskeni
```
> Not: Bu bilgiler sadece Docker konteynerinin, Chrome tarayıcısını sizin ekranınızda (Host) açabilmesi için gereklidir. Dışarıya gönderilmez.

### 2. Projeyi Başlatın

Terminali açın ve ana dizinde şu komutu çalıştırın. Bu işlem gerekli imajları oluşturacak ve servisleri ayağa kaldıracaktır.

```bash
docker compose up -d --build
```

### 3. Veri Kazıma ve Model Eğitimi (Airflow)
1.  **Airflow Arayüzüne** gidin: `http://localhost:8080`
2.  Listede `01_rfs_hybrid_pipeline` isimli iş akışını bulun.
3.  Sona erdiğinde çalışması için sol taraftaki **"Unpause"** (Anahtar) düğmesini açın.
4.  Sağ taraftaki **"Play"** butonuna basın ve **"Trigger DAG w/ config"** seçeneğine tıklayın.
5.  Açılan pencerede Linkleri değiştirmeden sadece sayfa sayılarını şu mantığa göre ayarlayın:

- **Hepsiburada:** Sayfa başına 36 ürün vardır. Max 50 sayfa seçin.

- **Trendyol:** Sayfa başına 16 ürün vardır. Max 100 sayfa seçin. > Neden? Bu oran (1:2), iki siteden de yaklaşık eşit sayıda ürün (Ortalama 1600-1800 adet) çekilmesini sağlayarak veri setini dengeli tutar.

6. "Trigger" butonuna basarak süreci başlatın.

### 3. İzleme ve Performans (Monitoring)

Süreç çalışırken arka planda neler olduğunu izleyebilirsiniz:

- **Loglar (Airflow):** Çalışan DAG'ın üzerine tıklayıp `Graph` görünümüne gelin. İlgili kutucuğa (Örn: `fetch_data`) tıklayıp **Logs** sekmesinden verilerin çekilişini canlı izleyebilirsiniz.

- **Model Performansı (MLflow):** Eğitim bittikten sonra http://localhost:5001 adresine gidin. Burada:

- Eğitilen tüm modellerin **RMSE**, **MAE** ve **R2** skorlarını karşılaştırabilirsiniz.

- **@champion** etiketi almış en iyi modeli görebilirsiniz.
---
## 🔗 Servis Erişim Adresleri

Proje çalıştığında aşağıdaki adreslerden servislere erişebilirsiniz:

| Servis | Adres | Açıklama |
| :--- | :--- | :--- |
| **🖥️ Kullanıcı Arayüzü** | **[http://localhost:5005](http://localhost:5005)** | Fiyat tahmini yapabileceğiniz ana ekran. |
| **⚙️ Model API (Swagger)** | **[http://localhost:8000/docs](http://localhost:8000/docs)** | API endpointlerini test edebileceğiniz panel. |
| **🧪 MLflow UI** | **[http://localhost:5001](http://localhost:5001)** | Model deneylerini, metrikleri ve parametreleri inceleyin. |
| **🌪️ Airflow UI** | **[http://localhost:8080](http://localhost:8080)** | İş akışlarını (DAGs) yönetin ve izleyin. |
| **📦 MinIO (S3)** | **[http://localhost:9001](http://localhost:9001)** | Kaydedilen model dosyalarını (Artifacts) görüntüleyin. |

*(Kullanıcı adı/şifre varsayılanları: `.env.exampe` dosyasında bulunabilır lütfen ismini `.env` olarak güncelleyiniz.)*

---

## 🛠️ Geliştirme Notları

* **Yeni Özellik Ekleme:** `configs/model_config.yaml` dosyasından modele girecek feature'ları açıp kapatabilirsiniz.
* **UI Güncellemesi:** `ui/templates/index.html` dosyasında yapılan değişiklikler için `docker compose restart ui` yeterlidir.
* **Model Seçenekleri:** Veri setine yeni bir marka eklendiğinde Airflow DAG'ını tekrar çalıştırmanız yeterlidir. UI otomatik güncellenir.