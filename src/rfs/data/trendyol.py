# Trendyol Scraper Class
import pandas as pd
from datetime import datetime
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    ElementClickInterceptedException,
    StaleElementReferenceException,
)

from src.rfs.data.base_scraper import BaseScraper


class TrendyolScraper(BaseScraper):
    def __init__(self):
        super().__init__(platform_name="TY")

        self.TARGET_FIELDS = [
            "Başlık",
            "Marka",
            "Kullanım Amacı",
            "Renk",
            "Cihaz Ağırlığı",
            "İşlemci Tipi",
            "İşlemci Modeli",
            "İşlemci Nesli",
            "İşlemci Çekirdek Sayısı",
            "Maksimum İşlemci Hızı (GHz)",
            "Ram (Sistem Belleği)",
            "Ram (Sistem Belleği) Tipi",
            "Ekran Kartı",
            "Ekran Kartı Tipi",
            "Ekran Kartı Hafızası",
            "Ekran Kartı Bellek Tipi",
            "SSD Kapasitesi",
            "Hard Disk Kapasitesi",
            "Ekran Boyutu",
            "Çözünürlük",
            "Çözünürlük Standartı",
            "Ekran Yenileme Hızı",
            "Panel Tipi",
            "İşletim Sistemi",
        ]

    def _expand_attributes(self, timeout: int = 10):
        """'Daha Fazla Göster' butonuna tıklayarak tüm özellikleri açar."""
        wait = WebDriverWait(self.driver, timeout)
        root_selector = "div.product-attributes-container.product-attributes, div[data-drroot='product-attributes']"

        try:
            root = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, root_selector))
            )
            self.driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center'});", root
            )

            buttons = root.find_elements(
                By.CSS_SELECTOR, ".show-more-section button.show-more-button"
            )
            if buttons:
                btn = buttons[0]
                self.driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'center'});", btn
                )
                try:
                    btn.click()
                except (
                    ElementClickInterceptedException,
                    StaleElementReferenceException,
                ):
                    self.driver.execute_script("arguments[0].click();", btn)

                # Genişlemenin tamamlanmasını bekle (basit heuristic)
                try:
                    wait.until(
                        lambda d: len(
                            d.find_elements(
                                By.CSS_SELECTOR, "div.attributes div.attribute-item"
                            )
                        )
                        > 12
                    )
                except Exception:
                    pass
        except TimeoutException:
            self.logger.warning("'Daha Fazla Göster' alanı bulunamadı.")

    def scrape_links(self, base_url: str, total_pages: int) -> pd.DataFrame:
        if not self.driver:
            self.start_driver()

        all_data = []
        for page in range(1, total_pages + 1):
            url = f"{base_url}&pi={page}"
            self.logger.info(f"Link toplama: Sayfa {page} işleniyor ({url})...")

            try:
                self.driver.get(url)
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_all_elements_located(
                        (By.CSS_SELECTOR, "a.product-card")
                    )
                )

                cards = self.driver.find_elements(By.CSS_SELECTOR, "a.product-card")
                for card in cards:
                    try:
                        brand = card.find_element(
                            By.CLASS_NAME, "product-brand"
                        ).text.strip()
                        name = card.find_element(
                            By.CLASS_NAME, "product-name"
                        ).text.strip()
                        price_elem = card.find_element(
                            By.CSS_SELECTOR, 'div[data-testid="single-price"]'
                        )

                        all_data.append(
                            {
                                "Name": f"{brand} {name}".strip(),
                                "Price": price_elem.text.strip(),
                                "Link": card.get_attribute("href"),
                                "Timestamp": datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                            }
                        )
                    except Exception:
                        continue
            except Exception as e:
                self.logger.error(f"Sayfa {page} hatası: {e}")

        df = pd.DataFrame(all_data)
        self.save_data(df, "Links", sub_folder="links")
        return df

    def _extract_single_product(self, link: str) -> dict:
        features = {field: None for field in self.TARGET_FIELDS}
        try:
            self.driver.get(link)
            wait = WebDriverWait(self.driver, 15)

            # Başlık ve Marka
            try:
                h1 = wait.until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "h1.product-title")
                    )
                )
                brand_text = h1.find_element(By.CSS_SELECTOR, "a strong").text.strip()
                features["Marka"] = brand_text
                features["Başlık"] = h1.text.strip().replace(brand_text, "").strip()
            except Exception:
                pass

            # Özellikleri Genişlet
            self._expand_attributes()

            # Verileri Oku
            attr_items = self.driver.find_elements(
                By.CSS_SELECTOR, "div.attributes div.attribute-item"
            )
            for item in attr_items:
                try:
                    label = item.find_element(By.CSS_SELECTOR, ".name").text.strip()
                    value = item.find_element(By.CSS_SELECTOR, ".value").text.strip()

                    if label in features:
                        if features[label] and features[label] != value:
                            features[label] = f"{features[label]}; {value}"
                        else:
                            features[label] = value
                except Exception:
                    continue

        except Exception as e:
            self.logger.error(f"Ürün hatası {link}: {e}")

        features["Çekilme Zamanı"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return features

    def scrape_details(self, links_df: pd.DataFrame) -> pd.DataFrame:
        if not self.driver:
            self.start_driver()

        results = []
        total = len(links_df)

        for i, (link, price) in enumerate(zip(links_df["Link"], links_df["Price"]), 1):
            if i % 10 == 0:
                self.logger.info(f"Detay toplama: {i}/{total} tamamlandı.")

            details = self._extract_single_product(link)
            details["Fiyat (TRY)"] = price
            details["Link"] = link
            results.append(details)

        df = pd.DataFrame(results)
        self.save_data(df, "Details", sub_folder="raw")
        return df
