from pathlib import Path

import pandas as pd

from mmm import config
from mmm.utils.data_cleaner import clean_dataset
from mmm.utils.logger import get_logger
from mmm.utils.validators import merged_data_schema

logger = get_logger(__name__)


class DataIngestion:
    """Handles loading, merging, cleaning, and validating marketing and sales data."""

    def __init__(self, data_dir: str = config.DATA_DIR):
        self.data_dir = Path(data_dir)

    def load_csv(self, filename: str) -> pd.DataFrame:
        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        logger.info(f"Loading {filename}")
        return pd.read_csv(file_path, parse_dates=["date"])

    def merge_all_data(self) -> pd.DataFrame:
        sales_df = self.load_csv("sales_data.csv")
        spend_files = {
            "tv_spend.csv": "tv_spend",
            "radio_spend.csv": "radio_spend",
            "social_media_spend.csv": "social_media_spend",
            "search_spend.csv": "search_spend",
            "print_spend.csv": "print_spend",
            "outdoor_spend.csv": "outdoor_spend",
        }

        merged_df = sales_df
        for file, col_name in spend_files.items():
            df = self.load_csv(file)
            if "spend" in df.columns:
                df = df.rename(columns={"spend": col_name})
            merged_df = pd.merge(merged_df, df, on="date", how="left")

        merged_df.sort_values("date", inplace=True)
        return merged_df

    def load_all_data(self) -> pd.DataFrame:
        merged_df = self.merge_all_data()
        merged_df = clean_dataset(merged_df)  #  Clean before validation
        merged_data_schema.validate(merged_df)  #  Final schema validation
        logger.info(f"Final dataset shape: {merged_df.shape}")
        return merged_df
