import pandas as pd

from mmm.utils.logger import get_logger

logger = get_logger(__name__)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the merged dataset by handling missing and invalid values.
    """
    # Drop rows with missing sales
    missing_sales = df["sales"].isna().sum()
    if missing_sales > 0:
        logger.warning(f"Dropping {missing_sales} rows with missing sales.")
        df = df.dropna(subset=["sales"])

    # Replace missing spend with 0
    spend_cols = [c for c in df.columns if c.endswith("_spend")]
    # explicitly assigns to a copy-safe DataFrame.
    df.loc[:, spend_cols] = df[spend_cols].fillna(0)

    # Remove negative sales
    negative_sales = (df["sales"] < 0).sum()
    if negative_sales > 0:
        logger.warning(f"Dropping {negative_sales} rows with negative sales.")
        df = df[df["sales"] >= 0]

    # Fix negative spend values
    for col in spend_cols:
        neg_values = (df[col] < 0).sum()
        if neg_values > 0:
            logger.warning(f"Fixing {neg_values} negative values in {col} to 0.")
            df.loc[df[col] < 0, col] = 0

    return df
