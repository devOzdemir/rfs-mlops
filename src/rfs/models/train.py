import json
import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
import optuna
import logging
from mlflow.tracking import MlflowClient  # <-- YENİ EKLENEN IMPORT
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Modeller
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Kendi modüllerimiz
from src.rfs.models.data_loader import load_training_data
from src.rfs.models.preprocessing import create_preprocessor
from src.rfs.utils.config_loader import load_config, setup_env_and_logging

# --- GLOBAL LOGGING AYARLARI ---
setup_env_and_logging()
logger = logging.getLogger("rfs_trainer")


class IndustrialTrainer:
    def __init__(self):
        # 1. Config Dosyasını Yükle
        self.config = load_config("configs/model_config.yaml")

        # 2. MLflow Bağlantısını Kur
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.config["experiment_name"])

        # 3. Veri Ayarlarını Çek
        self.data_config = self.config["data_config"]
        self.target_col = self.data_config["target_col"]
        self.drop_cols = self.data_config["drop_cols"]

        self.X, self.y, self.cat_cols, self.num_cols = self._prepare_data()

    def _prepare_data(self):
        """Veriyi yükler ve YAML Config kurallarına göre hazırlar."""
        df = load_training_data()

        cols_to_drop = self.drop_cols + [self.target_col]
        X = df.drop(columns=cols_to_drop, errors="ignore")
        y = df[self.target_col]

        logger.info(f"Target Column: {self.target_col}")
        logger.info(f"Dropped Columns: {self.drop_cols}")

        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X.select_dtypes(include=["number"]).columns.tolist()

        return X, y, cat_cols, num_cols

    def _evaluate_and_log(self, pipeline, X_train, y_train, X_test, y_test, prefix=""):
        """Değerlendirme ve Loglama."""
        train_preds = pipeline.predict(X_train)
        test_preds = pipeline.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_mae = mean_absolute_error(y_test, test_preds)
        test_r2 = r2_score(y_test, test_preds)
        overfit_gap = test_rmse - train_rmse

        logger.info(
            f"[{prefix}] Test RMSE: {test_rmse:.0f}, Gap: {overfit_gap:.0f}, MAE: {test_mae:.0f}"
        )

        metrics = {
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "test_r2": test_r2,
            "train_rmse": train_rmse,
            "overfit_gap": overfit_gap,
        }
        mlflow.log_metrics(metrics)
        return metrics

    def run_benchmark(self):
        """Tüm modelleri yarıştırır ve EN İYİSİNİN ADINI döndürür."""
        logger.info("Starting Benchmarking Mode...")

        model_factory = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(random_state=42),
            "LightGBM": LGBMRegressor(random_state=42, verbose=-1),
            "GradientBoosting": GradientBoostingRegressor(random_state=42),
        }

        models_to_run = self.config["benchmark_models"]
        best_model_name = None
        best_rmse = float("inf")

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        for model_name in models_to_run:
            if model_name not in model_factory:
                continue

            with mlflow.start_run(run_name=f"Benchmark_{model_name}"):
                mlflow.log_param("target_col", self.target_col)
                mlflow.log_param("dropped_cols", str(self.drop_cols))

                pipeline = Pipeline(
                    steps=[
                        (
                            "preprocessor",
                            create_preprocessor(self.cat_cols, self.num_cols),
                        ),
                        ("model", model_factory[model_name]),
                    ]
                )

                pipeline.fit(X_train, y_train)

                metrics = self._evaluate_and_log(
                    pipeline, X_train, y_train, X_test, y_test, prefix=model_name
                )
                mlflow.log_param("mode", "benchmark")

                signature = mlflow.models.infer_signature(
                    X_test, pipeline.predict(X_test)
                )
                mlflow.sklearn.log_model(pipeline, "model", signature=signature)

                if metrics["test_rmse"] < best_rmse:
                    best_rmse = metrics["test_rmse"]
                    best_model_name = model_name

        logger.info(
            f"🏆 BENCHMARK WINNER: {best_model_name} with RMSE: {best_rmse:.2f}"
        )
        return best_model_name

    def optimize_champion(self, model_name):
        """Seçilen şampiyon modeli dinamik olarak optimize eder."""
        logger.info(f"🚀 Starting Optimization for Champion: {model_name}")

        if model_name not in self.config["optuna"]:
            logger.warning(f"{model_name} için YAML'da optimizasyon parametresi yok.")
            return

        def objective(trial):
            model_params = self.config["optuna"][model_name]

            # (Parametre aralıkları aynı kalıyor - okunabilirlik için kısaltmadım)
            if model_name == "XGBoost":
                params = {
                    "n_estimators": trial.suggest_int(
                        "n_estimators",
                        model_params["n_estimators"]["low"],
                        model_params["n_estimators"]["high"],
                    ),
                    "max_depth": trial.suggest_int(
                        "max_depth",
                        model_params["max_depth"]["low"],
                        model_params["max_depth"]["high"],
                    ),
                    "learning_rate": trial.suggest_float(
                        "learning_rate",
                        model_params["learning_rate"]["low"],
                        model_params["learning_rate"]["high"],
                    ),
                    "subsample": trial.suggest_float(
                        "subsample",
                        model_params["subsample"]["low"],
                        model_params["subsample"]["high"],
                    ),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree",
                        model_params["colsample_bytree"]["low"],
                        model_params["colsample_bytree"]["high"],
                    ),
                    "random_state": 42,
                }
                model = XGBRegressor(**params)

            # ... Diğer modellerin parametre blokları (önceki kodun aynısı) ...
            # Buraya kodun kısalmaması için diğer modelleri eklemen gerekirse
            # önceki versiyondan kopyalayabilirsin, değişen bir şey yok.
            elif model_name == "LightGBM":
                params = {
                    "n_estimators": trial.suggest_int(
                        "n_estimators",
                        model_params["n_estimators"]["low"],
                        model_params["n_estimators"]["high"],
                    ),
                    "num_leaves": trial.suggest_int(
                        "num_leaves",
                        model_params["num_leaves"]["low"],
                        model_params["num_leaves"]["high"],
                    ),
                    "learning_rate": trial.suggest_float(
                        "learning_rate",
                        model_params["learning_rate"]["low"],
                        model_params["learning_rate"]["high"],
                    ),
                    "feature_fraction": trial.suggest_float(
                        "feature_fraction",
                        model_params["feature_fraction"]["low"],
                        model_params["feature_fraction"]["high"],
                    ),
                    "random_state": 42,
                    "verbose": -1,
                }
                model = LGBMRegressor(**params)
            elif model_name == "RandomForest":
                params = {
                    "n_estimators": trial.suggest_int(
                        "n_estimators",
                        model_params["n_estimators"]["low"],
                        model_params["n_estimators"]["high"],
                    ),
                    "max_depth": trial.suggest_int(
                        "max_depth",
                        model_params["max_depth"]["low"],
                        model_params["max_depth"]["high"],
                    ),
                    "min_samples_split": trial.suggest_int(
                        "min_samples_split",
                        model_params["min_samples_split"]["low"],
                        model_params["min_samples_split"]["high"],
                    ),
                    "random_state": 42,
                }
                model = RandomForestRegressor(**params)
            elif model_name == "ExtraTrees":
                params = {
                    "n_estimators": trial.suggest_int(
                        "n_estimators",
                        model_params["n_estimators"]["low"],
                        model_params["n_estimators"]["high"],
                    ),
                    "max_depth": trial.suggest_int(
                        "max_depth",
                        model_params["max_depth"]["low"],
                        model_params["max_depth"]["high"],
                    ),
                    "min_samples_split": trial.suggest_int(
                        "min_samples_split",
                        model_params["min_samples_split"]["low"],
                        model_params["min_samples_split"]["high"],
                    ),
                    "random_state": 42,
                }
                model = ExtraTreesRegressor(**params)
            elif model_name == "GradientBoosting":
                params = {
                    "n_estimators": trial.suggest_int(
                        "n_estimators",
                        model_params["n_estimators"]["low"],
                        model_params["n_estimators"]["high"],
                    ),
                    "learning_rate": trial.suggest_float(
                        "learning_rate",
                        model_params["learning_rate"]["low"],
                        model_params["learning_rate"]["high"],
                    ),
                    "max_depth": trial.suggest_int(
                        "max_depth",
                        model_params["max_depth"]["low"],
                        model_params["max_depth"]["high"],
                    ),
                    "subsample": trial.suggest_float(
                        "subsample",
                        model_params["subsample"]["low"],
                        model_params["subsample"]["high"],
                    ),
                    "min_samples_split": trial.suggest_int(
                        "min_samples_split",
                        model_params["min_samples_split"]["low"],
                        model_params["min_samples_split"]["high"],
                    ),
                    "random_state": 42,
                }
                model = GradientBoostingRegressor(**params)
            elif model_name == "Ridge":
                params = {
                    "alpha": trial.suggest_float(
                        "alpha",
                        model_params["alpha"]["low"],
                        model_params["alpha"]["high"],
                        log=True,
                    )
                }
                model = Ridge(**params)
            elif model_name == "Lasso":
                params = {
                    "alpha": trial.suggest_float(
                        "alpha",
                        model_params["alpha"]["low"],
                        model_params["alpha"]["high"],
                        log=True,
                    )
                }
                model = Lasso(**params)
            else:
                return float("inf")

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", create_preprocessor(self.cat_cols, self.num_cols)),
                    ("model", model),
                ]
            )

            scores = cross_val_score(
                pipeline, self.X, self.y, cv=3, scoring="neg_mean_squared_error"
            )
            return np.sqrt(-scores.mean())

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.config["optuna"]["n_trials"])

        logger.info(f"Best Params for {model_name}: {study.best_params}")
        self._train_and_log_best_model(model_name, study.best_params)

    def _save_categorical_options(self):
        """Kategorik kolonlardaki benzersiz değerleri JSON olarak kaydeder."""
        options = {}
        # Sadece eğitimde kullanılan kategorik kolonları al
        for col in self.cat_cols:
            # unique değerleri al ve listeye çevir
            unique_vals = self.X[col].dropna().unique().tolist()
            options[col] = sorted(unique_vals)

        # Dosyayı kaydet
        with open("categorical_options.json", "w") as f:
            json.dump(options, f, ensure_ascii=False)

        # MLflow'a artifact olarak gönder
        mlflow.log_artifact("categorical_options.json")
        logger.info("✅ Kategorik seçenekler (metadata) MLflow'a kaydedildi.")

    def _train_and_log_best_model(self, model_name, params):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        with mlflow.start_run(run_name=f"Champion_{model_name}_Optimized"):
            mlflow.log_param("target_col", self.target_col)
            mlflow.log_params(params)

            if model_name == "XGBoost":
                model = XGBRegressor(**params)
            elif model_name == "LightGBM":
                model = LGBMRegressor(**params, verbose=-1)
            elif model_name == "RandomForest":
                model = RandomForestRegressor(**params)
            elif model_name == "ExtraTrees":
                model = ExtraTreesRegressor(**params)
            elif model_name == "GradientBoosting":
                model = GradientBoostingRegressor(**params)
            elif model_name == "Ridge":
                model = Ridge(**params)
            elif model_name == "Lasso":
                model = Lasso(**params)
            else:
                return

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", create_preprocessor(self.cat_cols, self.num_cols)),
                    ("model", model),
                ]
            )

            pipeline.fit(X_train, y_train)

            self._evaluate_and_log(
                pipeline, X_train, y_train, X_test, y_test, prefix="Champion"
            )
            # --- YENİ: Metadata Kaydı ---
            self._save_categorical_options()

            signature = mlflow.models.infer_signature(X_test, pipeline.predict(X_test))

            # --- 1. MODELİ KAYDET (REGISTER) ---
            # Bu işlem modeli 'RFS_Laptop_Price_Predictor' adıyla kaydeder (v1, v2...)
            model_info = mlflow.sklearn.log_model(
                pipeline,
                "model",
                signature=signature,
                registered_model_name="RFS_Laptop_Price_Predictor",
            )

            # --- 2. ALIAS ATAMA (ETİKETLEME) ---
            # Kaydedilen son versiyonu alıp ona '@champion' etiketini yapıştırıyoruz.
            # API (FastAPI) bu etiketi takip edecek.
            client = MlflowClient()
            model_version = model_info.registered_model_version
            client.set_registered_model_alias(
                name="RFS_Laptop_Price_Predictor",
                alias="champion",
                version=model_version,
            )

            logger.info(
                f"Champion Model Saved! Version: {model_version} -> Alias: @champion"
            )


if __name__ == "__main__":
    trainer = IndustrialTrainer()
    winner = trainer.run_benchmark()
    if winner:
        trainer.optimize_champion(winner)
