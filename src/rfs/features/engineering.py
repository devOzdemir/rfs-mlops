# Feature engineering (PPI, etc.)
import pandas as pd
import numpy as np
import re


def calculate_ppi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pixel Per Inch (PPI) değerini hesaplar.
    Formül: sqrt(w^2 + h^2) / screen_inch
    """
    df = df.copy()

    # Geçici listeler
    widths = []
    heights = []

    for res in df["resolution"]:
        if pd.isna(res):
            widths.append(np.nan)
            heights.append(np.nan)
            continue

        # "1920x1080" formatını parse et
        try:
            parts = str(res).lower().split("x")
            if len(parts) == 2:
                widths.append(float(parts[0]))
                heights.append(float(parts[1]))
            else:
                widths.append(np.nan)
                heights.append(np.nan)
        except (ValueError, TypeError):  # Hatalı format
            widths.append(np.nan)
            heights.append(np.nan)

    # Hesaplama
    df["res_w"] = widths
    df["res_h"] = heights

    # PPI Formülü
    # screen_size_inch 0 veya Nan ise hata vermemesi için kontrol
    df["ppi"] = np.sqrt(df["res_w"] ** 2 + df["res_h"] ** 2) / df["screen_size_inch"]

    # Sonsuz değerleri (inf) veya saçma değerleri temizle
    df["ppi"] = df["ppi"].replace([np.inf, -np.inf], np.nan)

    # Gereksiz geçici kolonları at
    df = df.drop(columns=["res_w", "res_h"])

    return df


def group_rare_categories(
    df: pd.DataFrame, col: str, threshold: int = 10
) -> pd.DataFrame:
    """
    Az frekanslı kategorileri 'Other' altında toplar.
    Örn: Veri setinde sadece 1 tane 'Fujitsu' varsa, model bunu öğrenemez. 'Other' yaparız.
    """
    if col not in df.columns:
        return df

    counts = df[col].value_counts()
    rare_labels = counts[counts < threshold].index

    if len(rare_labels) > 0:
        df[col] = df[col].apply(lambda x: "Other" if x in rare_labels else x)

    return df


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Tüm mühendislik işlemlerini uygular."""

    # 1. PPI Hesapla (En kritik feature)
    df = calculate_ppi(df)

    # 2. Marka Gruplama (Çok nadir markalar gürültü yaratır)
    # Eşik değeri veri büyüklüğüne göre değişir, şimdilik 5 diyelim.
    df = group_rare_categories(df, "brand", threshold=5)

    # 3. İşletim Sistemi Gruplama
    df = group_rare_categories(df, "operating_system", threshold=5)

    # 4. Eksik PPI ve Fiyatları Temizle (Model için olmazsa olmazlar)
    df = df.dropna(subset=["ppi", "price_try"])

    return df
