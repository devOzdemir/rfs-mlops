import pandas as pd
from datetime import datetime
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException

# Base Class import
from src.rfs.data.base_scraper import BaseScraper


class HepsiburadaScraper(BaseScraper):
    def __init__(self):
        super().__init__(platform_name="HB")

        self.TARGET_FIELDS = [
            "Başlık",
            "Marka",
            "Kullanım Amacı",
            "Renk",
            "Cihaz Ağırlığı",
            "İşlemci Tipi",
            "İşlemci",
            "İşlemci Nesli",
            "İşlemci Çekirdek Sayısı",
            "Maksimum İşlemci Hızı",
            "Ram (Sistem Belleği)",
            "Ram Tipi",
            "Ekran Kartı",
            "Ekran Kartı Tipi",
            "Ekran Kartı Hafızası",
            "Ekran Kartı Bellek Tipi",
            "SSD Kapasitesi",
            "Harddisk Kapasitesi",
            "Ekran Boyutu",
            "Max Ekran Çözünürlüğü",
            "Ekran Özelliği",
            "Ekran Yenileme Hızı",
            "Ekran Panel Tipi",
            "İşletim Sistemi",
        ]

    def _wait_dom_interactive(self, timeout: int = 10) -> None:
        WebDriverWait(self.driver, timeout).until(
            lambda d: d.execute_script("return document.readyState")
            in ("interactive", "complete")
        )

    def _scroll_to_tech_specs(self, timeout: int = 20):
        self._wait_dom_interactive(timeout=min(10, timeout))
        wait = WebDriverWait(self.driver, timeout, poll_frequency=0.35)

        def _try_jump(d):
            try:
                candidates = d.find_elements(By.CSS_SELECTOR, "a[href='#techSpecs']")
                candidates.extend(
                    d.find_elements(
                        By.XPATH,
                        "//a[contains(., 'Teknik') and contains(., 'Özellik')]",
                    )
                )
                if candidates:
                    el = candidates[0]
                    d.execute_script(
                        "arguments[0].scrollIntoView({block: 'center'});", el
                    )
                    d.execute_script("arguments[0].click();", el)
            except Exception:
                pass

        self.driver.execute_script("window.scrollTo(0, 250);")

        def _cond(d):
            try:
                el = d.find_element(By.ID, "techSpecs")
                d.execute_script("arguments[0].scrollIntoView({block: 'center'});", el)
                return el
            except NoSuchElementException:
                _try_jump(d)
                d.execute_script("window.scrollBy(0, 350);")
                return False

        try:
            return wait.until(_cond)
        except Exception:
            return None

    def scrape_links(self, base_url: str, total_pages: int) -> pd.DataFrame:
        if not self.driver:
            self.start_driver()

        all_results = []
        for page in range(1, total_pages + 1):
            self.logger.info(f"Link toplama: Sayfa {page} işleniyor...")
            self.driver.get(base_url + str(page))

            try:
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "[data-test-id^='final-price']")
                    )
                )
                items = self.driver.find_elements(By.TAG_NAME, "li")

                for item in items:
                    try:
                        a_tag = item.find_element(By.TAG_NAME, "a")
                        price_tag = item.find_element(
                            By.CSS_SELECTOR, "[data-test-id^='final-price']"
                        )

                        all_results.append(
                            {
                                "Name": a_tag.get_attribute("title"),
                                "Price": price_tag.text.replace("\n", " ").strip(),
                                "Link": a_tag.get_attribute("href"),
                            }
                        )
                    except Exception:
                        continue
            except Exception as e:
                self.logger.error(f"Sayfa {page} yüklenirken hata: {e}")

        df = pd.DataFrame(all_results)

        # --- KRİTİK NOKTA: LİNKLERİ KAYDET ---
        # Bu satır veriyi links şemasına yazar.
        self.save_data(df, "Links", sub_folder="links")

        return df

    def _extract_single_product(self, link: str) -> dict:
        features = {field: None for field in self.TARGET_FIELDS}
        try:
            self.driver.get(link)
            self._wait_dom_interactive()
            wait = WebDriverWait(self.driver, 15)

            try:
                title = wait.until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, '[data-test-id="title"]')
                    )
                )
                brand = wait.until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, '[data-test-id="brand"]')
                    )
                )
                features["Başlık"] = title.text.strip()
                features["Marka"] = brand.get_attribute("title").strip()
            except Exception:
                self.logger.warning(f"Başlık/Marka alınamadı: {link}")

            tech_specs = self._scroll_to_tech_specs()
            if tech_specs:
                rows = tech_specs.find_elements(By.CLASS_NAME, "jkj4C4LML4qv2Iq8GkL3")
                for row in rows:
                    try:
                        label = row.find_element(
                            By.CLASS_NAME, "OXP5AzPvafgN_i3y6wGp"
                        ).text.strip()
                        val_el = row.find_element(By.CLASS_NAME, "AxM3TmSghcDRH1F871Vh")
                        links = val_el.find_elements(By.TAG_NAME, "a")
                        value = (
                            links[0].get_attribute("title").strip()
                            if links
                            else val_el.text.strip()
                        )

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

        # Detayları kaydet -> raw şemasına
        self.save_data(df, "Details", sub_folder="raw")

        return df
