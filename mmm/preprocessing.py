from typing import List

import numpy as np
import pandas as pd

from mmm import config
from mmm.utils.logger import get_logger

logger = get_logger(__name__)


def apply_adstock(spend: np.ndarray, decay: float) -> np.ndarray:
    """
    Apply adstock transformation to a spend series.
    decay: value between 0 and 1 (higher → longer carryover)
    """
    result = np.zeros_like(spend)
    for t in range(len(spend)):
        if t == 0:
            result[t] = spend[t]
        else:
            result[t] = spend[t] + decay * result[t - 1]
    return result


def apply_saturation(spend: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """
    Apply saturation transformation (Hill function).
    alpha: controls scaling
    gamma: controls curvature (diminishing returns)
    """
    return alpha * (spend**gamma) / (spend**gamma + alpha**gamma)


class Preprocessor:
    """
    Handles feature engineering for MMM (adstock + saturation).
    """

    def __init__(self, decay_factors=None, alpha_params=None, gamma_params=None):
        self.decay_factors = decay_factors or config.DEFAULT_DECAY_FACTORS
        self.alpha_params = alpha_params or config.DEFAULT_ALPHA_PARAMS
        self.gamma_params = gamma_params or config.DEFAULT_GAMMA_PARAMS

    def transform(self, df: pd.DataFrame, spend_cols: List[str]) -> pd.DataFrame:
        transformed_df = df.copy()
        for col in spend_cols:
            logger.info(f"Applying adstock and saturation to {col}")

            # Adstock
            adstocked = apply_adstock(
                transformed_df[col].values, self.decay_factors.get(col, 0.5)
            )

            # Saturation
            saturated = apply_saturation(
                adstocked,
                self.alpha_params.get(col, 1.0),
                self.gamma_params.get(col, 1.0),
            )

            transformed_df[f"{col}_transformed"] = saturated

        return transformed_df
