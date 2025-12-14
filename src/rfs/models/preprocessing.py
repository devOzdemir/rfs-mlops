import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def create_preprocessor(categorical_cols: list, numerical_cols: list):
    """
    Veri ön işleme pipeline'ını oluşturur.
    - Kategorik: Eksik veriyi 'missing' ile doldur -> OneHotEncode
    - Sayısal: Eksik veriyi 'median' ile doldur -> StandartScaler
    """

    # 1. Sayısal İşlemler
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),  # Eksikleri medyan ile doldur
            ("scaler", StandardScaler()),  # 0-1 arasına çek (Normalize et)
        ]
    )

    # 2. Kategorik İşlemler
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),  # Eksikleri 'missing' diye etiketle
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),  # Bilinmeyen kategori gelirse hata verme (ignore)
        ]
    )

    # 3. Birleştirme
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        verbose_feature_names_out=False,  # Kolon isimlerini temiz tut
    )

    return preprocessor
