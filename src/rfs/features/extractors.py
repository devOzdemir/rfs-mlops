import re
import pandas as pd
from typing import Optional, Union

# -----------------------------
# Regex Patterns
# -----------------------------
REGEX_PATTERNS = {
    "GB_TOKEN": re.compile(r"(\d{1,4})\s*gb", re.IGNORECASE),
    "HZ_TOKEN": re.compile(r"(\d{1,3})\s*hz", re.IGNORECASE),
    "SSD_GB": re.compile(
        r"(?<!\d)(\d{1,4})\s*gb\s*(?:nvme\s*)?(?:m\.?2\s*)?(?:ssd|gbssd)\b",
        re.IGNORECASE,
    ),
    "SSD_TB": re.compile(
        r"(?<!\d)(\d+(?:\.\d+)?)\s*tb\s*(?:nvme\s*)?(?:m\.?2\s*)?(?:ssd|gbssd)\b",
        re.IGNORECASE,
    ),
    "RES": re.compile(r"(\d{3,4})\s*[x×X]\s*(\d{3,4})"),
}

# Hepsiburada dağılımına göre izinli SSD kapasiteleri
ALLOWED_SSD_GB = {4, 120, 128, 250, 256, 500, 512, 1024, 2048, 4096, 8192}

# Ekran Çözünürlük Haritası
RES_MAP = {
    (1920, 1080): "Full HD",
    (1920, 1200): "WUXGA",
    (2560, 1600): "WQXGA",
    (2560, 1440): "QHD",
    (2880, 1800): "QHD+",
    (3200, 2000): "QHD+",
}


def _soft_clean(s: str) -> str:
    """Temel metin temizliği."""
    s = str(s).lower()
    s = re.sub(r"[-_/|]+", " ", s)
    s = s.replace(",", ".")
    return re.sub(r"\s+", " ", s).strip()


def extract_ssd_from_title(title: str) -> Optional[int]:
    """Başlıktan SSD kapasitesini GB cinsinden çıkarır."""
    t = _soft_clean(title)
    cands = []

    for m in REGEX_PATTERNS["SSD_TB"].finditer(t):
        cands.append(int(round(float(m.group(1)) * 1024)))

    for m in REGEX_PATTERNS["SSD_GB"].finditer(t):
        cands.append(int(m.group(1)))

    if not cands:
        return None

    val = max(cands)
    return val if val in ALLOWED_SSD_GB else None


def extract_ram_from_title(title: str) -> Optional[float]:
    """Başlıktan RAM kapasitesini çıkarır. SSD kapasitesi ile karışmaması için mantıksal kontrol yapar."""
    low = str(title).strip().lower()
    tokens = [
        (int(m.group(1)), m.start(), m.end())
        for m in REGEX_PATTERNS["GB_TOKEN"].finditer(low)
    ]

    if not tokens:
        return None

    ssd_gb = extract_ssd_from_title(title)
    ssd_pos = low.find("ssd")

    for val, _, en in tokens:
        # RAM genellikle 128GB altındadır
        if 0 < val <= 128:
            # Eğer SSD değeri ile aynıysa ve SSD kelimesinden önce geliyorsa dikkat et
            if ssd_gb is not None and val == ssd_gb:
                continue
            # SSD kelimesinden önce gelen küçük GB değeri genellikle RAM'dir
            if ssd_pos != -1 and en < ssd_pos:
                return float(val)
            # SSD yoksa ilk makul değeri RAM kabul et
            if ssd_pos == -1:
                return float(val)
    return None


def extract_refresh_rate_from_title(title: str) -> Optional[float]:
    """Başlıktan Hz değerini çıkarır."""
    vals = [int(m.group(1)) for m in REGEX_PATTERNS["HZ_TOKEN"].finditer(str(title))]
    vals = [v for v in vals if 30 <= v <= 360]
    return float(max(vals)) if vals else None


def extract_resolution_from_title(title: str) -> Optional[str]:
    """Başlıktan ekran özelliklerini (Retina, FHD vb.) çıkarır."""
    low = str(title).lower()

    if "liquid retina" in low or "retina" in low:
        return "Retina"

    priorities = ["OLED", "WQXGA", "WUXGA", "QHD+", "QHD", "Full HD", "FHD", "HD"]
    for feat in priorities:
        if feat.lower() in low:
            return feat

    m = REGEX_PATTERNS["RES"].search(str(title))
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        return RES_MAP.get((w, h), None)

    return None


def fill_missing_from_title(
    df: pd.DataFrame, source_col: str, target_col: str, extractor_func
) -> pd.DataFrame:
    """Hedef kolondaki eksik verileri, kaynak kolondan (title) çıkarılan veriyle doldurur."""
    mask = df[target_col].isna()
    extracted = df.loc[mask, source_col].apply(extractor_func)
    df.loc[mask, target_col] = extracted
    return df
