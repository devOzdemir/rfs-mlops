import sys
import os

# Proje kök dizinini path'e ekle (Garanti olsun diye)
sys.path.append(os.getcwd())

from src.rfs.features.pipeline import LaptopETLPipeline


def main():
    print("ETL Pipeline Başlatılıyor...")
    print("--------------------------------")

    try:
        # Pipeline nesnesini oluştur
        pipeline = LaptopETLPipeline()

        # Çalıştır
        pipeline.run()

        print("--------------------------------")
        print("ETL İşlemi Başarıyla Tamamlandı!")
        print("Şimdi 'transform.laptops' tablosunu sorgulayabilirsin.")

    except Exception as e:
        print(f"\n Kritik Hata Oluştu: {e}")
        # Detaylı hatayı görmek için raise edebiliriz
        raise


if __name__ == "__main__":
    main()
