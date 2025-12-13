import re
import numpy as np
import pandas as pd
from typing import Any, Optional


# --- Yardımcı Fonksiyonlar ---
def _clean_text(val: Any) -> Optional[str]:
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    invalids = {"", "nan", "none", "null", "-", "yok", "belirtilmemiş", "belirtilmemis"}
    return None if s in invalids else re.sub(r"\s+", " ", s)


# --- Numeric Parsers ---
def parse_numeric(val: Any, min_val: float, max_val: float) -> Optional[float]:
    if pd.isna(val):
        return None

    if isinstance(val, (int, float)):
        return float(val) if min_val <= val <= max_val else None

    # "16 GB" -> 16.0
    s = str(val).strip().lower().replace(",", ".")
    m = re.search(r"(\d+(\.\d+)?)", s)
    if m:
        try:
            v = float(m.group(1))
            return v if min_val <= v <= max_val else None
        except ValueError:
            pass
    return None


def parse_price(val: Any) -> Optional[float]:
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)

    # 1.299,00 TL veya 1.299 TL temizliği
    s = str(val).lower().replace("tl", "").replace("₺", "").strip()
    # Nokta binlik, virgül ondalık ise:
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif "." in s and len(s.split(".")[-1]) == 3:  # 45.000 gibi
        s = s.replace(".", "")
    else:
        s = s.replace(",", ".")

    try:
        price = float(s)
        return price if 1000 <= price <= 1_000_000 else None
    except ValueError:
        return None


# --- Categorical Parsers ---
def parse_brand(val: Any) -> Optional[str]:
    s = _clean_text(val)
    if not s:
        return None

    mapping = {
        "hewlett packard": "hp",
        "h.p.": "hp",
        "ilife": "i-life",
        "i-life digital": "i-life",
        "game garaj": "game garaj",
    }
    return mapping.get(s, s)  # Mapping yoksa kendisini döndür


def parse_os(val: Any) -> Optional[str]:
    s = _clean_text(val)
    if not s:
        return None

    if "freedos" in s or "yok" in s:
        return "freedos"
    if "mac" in s or "macos" in s:
        return "macos"
    if "android" in s:
        return "linux"  # Android laptopları linux sayabiliriz
    if "ubuntu" in s or "linux" in s:
        return "linux"
    if "windows" in s:
        return "windows pro" if "pro" in s else "windows home"
    return "unknown"


def parse_gpu_type(val: Any) -> Optional[str]:
    s = _clean_text(val)
    if not s:
        return None

    if any(x in s for x in ["dahili", "onboard", "integrated", "paylaşım"]):
        return "integrated"
    if "yüksek seviye" in s:
        return "dedicated_high_end"
    if "harici" in s:
        return "dedicated"
    return "unknown"


def parse_panel_type(val: Any) -> Optional[str]:
    s = _clean_text(val)
    if not s:
        return None

    # Sadece geçerli panel tiplerini kabul et
    valid_panels = ["ips", "tn", "va", "oled", "mini led", "led"]
    for p in valid_panels:
        if p in s:
            return p
    return None


def parse_resolution(val: Any) -> Optional[str]:
    s = _clean_text(val)
    if not s:
        return None
    s = s.replace("×", "x").replace("X", "x")
    m = re.search(r"(\d{3,5})\s*x\s*(\d{3,5})", s)
    return f"{m.group(1)}x{m.group(2)}" if m else None


def parse_cpu_family(val: Any) -> Optional[str]:
    s = _clean_text(val)
    if not s:
        return None

    if "apple" in s or " m1" in s or " m2" in s or " m3" in s:
        return "apple silicon"
    if "celeron" in s:
        return "intel celeron"
    if "ryzen" in s:
        m = re.search(r"ryzen\s*(3|5|7|9)", s)
        return f"amd ryzen {m.group(1)}" if m else "amd ryzen"
    if "core" in s:
        # Core Ultra
        if "ultra" in s:
            m = re.search(r"ultra\s*(5|7|9)", s)
            return f"intel core ultra {m.group(1)}" if m else "intel core ultra"
        # Core i Series
        m = re.search(r"i\s*([3579])", s)
        return f"intel core i{m.group(1)}" if m else "intel core"

    return "other"
