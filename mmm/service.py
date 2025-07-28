from pathlib import Path

import pandas as pd

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
        """
        Prepares X (and y if available) for training or prediction.
        training=True -> expects 'sales' column
        training=False -> only returns X
        """
        spend_cols = [c for c in df.columns if c.endswith("_spend")]
        transformed_df = self.preprocessor.transform(df, spend_cols)

        feature_cols = [f"{c}_transformed" for c in spend_cols]
        X = transformed_df[feature_cols]

        y = (
            transformed_df["sales"]
            if training and "sales" in transformed_df.columns
            else None
        )
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
        return pd.Series(preds, index=new_data.index)
