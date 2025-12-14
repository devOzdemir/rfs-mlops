from src.rfs.data.hepsiburada import HepsiburadaScraper
from src.rfs.data.trendyol import TrendyolScraper


def main():
    # Konfigürasyon
    CONFIG = {
        "HB": {
            "url": "https://www.hepsiburada.com/laptop-notebook-dizustu-bilgisayarlar-c-98?puan=3-max&sayfa=",
            "pages": 10,  # Test için düşük tutuldu
        },
        "TY": {
            "url": "https://www.trendyol.com/sr?wc=103108%2C106084&sst=MOST_RATED",
            "pages": 20,  # Test için düşük tutuldu
        },
    }

    # --- HEPSIBURADA ---
    print(">>> Hepsiburada süreci başlıyor...")
    hb_scraper = HepsiburadaScraper()
    try:
        # 1. Linkleri topla
        hb_links = hb_scraper.scrape_links(CONFIG["HB"]["url"], CONFIG["HB"]["pages"])

        # 2. Detayları topla
        if not hb_links.empty:
            hb_scraper.scrape_details(hb_links)
    finally:
        hb_scraper.close_driver()

    # --- TRENDYOL ---
    print("\n>>> Trendyol süreci başlıyor...")
    ty_scraper = TrendyolScraper()
    try:
        ty_links = ty_scraper.scrape_links(CONFIG["TY"]["url"], CONFIG["TY"]["pages"])

        if not ty_links.empty:
            ty_scraper.scrape_details(ty_links)
    finally:
        ty_scraper.close_driver()


if __name__ == "__main__":
    main()
