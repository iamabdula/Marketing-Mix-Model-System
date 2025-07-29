from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from mmm import config
from mmm.data_ingestion import DataIngestion
from mmm.models.factory import ModelFactory
from mmm.preprocessing import Preprocessor
from mmm.utils.logger import get_logger

logger = get_logger(__name__)


class MMMService:
    """
    Handles training, saving, loading, and predicting with a Marketing Mix Model.
    """

    def __init__(self, model_type: str = "linear", model_dir: str = config.MODEL_DIR):
        self.model_type = model_type
        self.model = ModelFactory.get_model(model_type)
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / f"{model_type}_mmm_model.pkl"

        self.data_ingestion = DataIngestion()
        self.preprocessor = Preprocessor()

        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_features(self, df: pd.DataFrame, training: bool = True):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        spend_cols = [c for c in df.columns if c.endswith("_spend")]
        transformed_df = self.preprocessor.transform(df, spend_cols)

        # Add lagged sales & rolling averages
        if "sales" in transformed_df.columns:
            transformed_df["sales_lag1"] = (
                transformed_df["sales"].shift(1).fillna(method="bfill")
            )
            transformed_df["sales_lag7"] = (
                transformed_df["sales"].shift(7).fillna(method="bfill")
            )
            transformed_df["sales_lag14"] = (
                transformed_df["sales"].shift(14).fillna(method="bfill")
            )
            transformed_df["sales_ma7"] = (
                transformed_df["sales"]
                .rolling(7)
                .mean()
                .shift(1)
                .fillna(method="bfill")
            )
            transformed_df["sales_ma14"] = (
                transformed_df["sales"]
                .rolling(14)
                .mean()
                .shift(1)
                .fillna(method="bfill")
            )
        else:
            for col in [
                "sales_lag1",
                "sales_lag7",
                "sales_lag14",
                "sales_ma7",
                "sales_ma14",
            ]:
                transformed_df[col] = 0

        # Add time features
        transformed_df["quarter"] = transformed_df["date"].dt.quarter
        transformed_df["weekofyear"] = transformed_df["date"].dt.isocalendar().week
        transformed_df["is_weekend"] = (
            transformed_df["date"].dt.dayofweek >= 5
        ).astype(int)

        # Select features
        feature_cols = [f"{c}_transformed" for c in spend_cols] + [
            "sales_lag1",
            "sales_lag7",
            "sales_lag14",
            "sales_ma7",
            "sales_ma14",
            "quarter",
            "weekofyear",
            "is_weekend",
        ]

        X = transformed_df[feature_cols].copy()
        X = np.log1p(X)  # log transform features

        # Target as log(sales)
        y = None
        if training and "sales" in transformed_df.columns:
            y = np.log1p(transformed_df["sales"])

        return X, y

    def train(self) -> None:
        """
        Loads processed data, trains the selected model, and saves it to disk.
        """
        logger.info(f"Training {self.model_type.upper()} MMM model...")

        df = self.data_ingestion.load_all_data()
        X, y = self._prepare_features(df, training=True)

        self.model.train(X, y)
        self.model.save(self.model_path)

        logger.info(f"Model trained and saved to {self.model_path}")

    def load_model(self) -> None:
        """
        Loads a trained model from disk.
        """
        logger.info(f"Loading {self.model_type.upper()} MMM model...")
        self.model.load(self.model_path)

    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        """
        Applies preprocessing to new spend data and returns predictions.
        """
        X, _ = self._prepare_features(new_data, training=False)
        preds = self.model.predict(X)
        preds = np.expm1(preds)  # Convert back from log scale
        return pd.Series(preds, index=new_data.index)

    def evaluate_models(self) -> dict:
        df = self.data_ingestion.load_all_data()
        X, y = self._prepare_features(df, training=True)

        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        results = {}

        for model_type in ["linear", "ridge", "rf"]:
            model = ModelFactory.get_model(model_type)
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = {
                "r2": float(r2_score(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mape": float(
                    np.mean(
                        np.abs(
                            (y_test.clip(lower=1e-6) - y_pred) / y_test.clip(lower=1e-6)
                        )
                    )
                ),
            }
            results[model_type] = metrics

        return results
