import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from mmm.models.base import MMMModel


class LinearMMM(MMMModel):
    """
    Linear Regression-based MMM model.
    """

    def __init__(self):
        self.model = LinearRegression()

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)


class RidgeMMM(MMMModel):
    """
    Ridge Regression-based MMM model.
    """

    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
