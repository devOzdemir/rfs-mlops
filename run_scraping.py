import argparse
import sys
from src.rfs.data.hepsiburada import HepsiburadaScraper
from src.rfs.data.trendyol import TrendyolScraper


def main():
    parser = argparse.ArgumentParser(description="RFS Scraper Runner")

    # Hangi site?
    parser.add_argument("--site", type=str, choices=["hb", "ty", "all"], default="all")

    # Dinamik Parametreler (Varsayılanlar yine kodda kalsın, boş gelirse patlamasın)
    parser.add_argument(
        "--hb-url",
        type=str,
        default="https://www.hepsiburada.com/laptop-notebook-dizustu-bilgisayarlar-c-98?puan=3-max&sayfa=",
    )
    parser.add_argument("--hb-pages", type=int, default=1)

    parser.add_argument(
        "--ty-url",
        type=str,
        default="https://www.trendyol.com/sr?wc=103108%2C106084&sst=MOST_RATED",
    )
    parser.add_argument("--ty-pages", type=int, default=1)

    args = parser.parse_args()

    # Konfigürasyonu Argümanlardan Oluştur
    CONFIG = {
        "HB": {
            "url": args.hb_url,
            "pages": args.hb_pages,
        },
        "TY": {
            "url": args.ty_url,
            "pages": args.ty_pages,
        },
    }

    # --- HEPSIBURADA ---
    if args.site in ["hb", "all"]:
        print(f">>> Hepsiburada başlıyor... (Sayfa: {CONFIG['HB']['pages']})")
        hb_scraper = HepsiburadaScraper()
        try:
            hb_links = hb_scraper.scrape_links(
                CONFIG["HB"]["url"], CONFIG["HB"]["pages"]
            )
            if not hb_links.empty:
                hb_scraper.scrape_details(hb_links)
        except Exception as e:
            print(f"HB Hatası: {e}")
            if args.site == "hb":
                sys.exit(1)
        finally:
            hb_scraper.close_driver()

    # --- TRENDYOL ---
    if args.site in ["ty", "all"]:
        print(f"\n>>> Trendyol başlıyor... (Sayfa: {CONFIG['TY']['pages']})")
        ty_scraper = TrendyolScraper()
        try:
            ty_links = ty_scraper.scrape_links(
                CONFIG["TY"]["url"], CONFIG["TY"]["pages"]
            )
            if not ty_links.empty:
                ty_scraper.scrape_details(ty_links)
        except Exception as e:
            print(f"TY Hatası: {e}")
            if args.site == "ty":
                sys.exit(1)
        finally:
            ty_scraper.close_driver()


if __name__ == "__main__":
    main()
