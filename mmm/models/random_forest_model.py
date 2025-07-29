import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from mmm.models.base import MMMModel


class RandomForestMMM(MMMModel):
    def __init__(self):
        self.model = RandomForestRegressor(random_state=42)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        # ✅ Hyperparameter tuning
        param_grid = {
            "n_estimators": [200, 500],
            "max_depth": [5, 10],
            "min_samples_leaf": [5, 10],
        }
        grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=3,
            scoring="r2",
            n_jobs=-1,
        )
        grid.fit(X, y)
        self.model = grid.best_estimator_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
