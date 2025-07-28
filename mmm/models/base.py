from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class MMMModel(ABC):
    """
    Abstract Base Class for all MMM models.
    Enforces a consistent interface for training, prediction, saving, and loading.
    """

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass
