from mmm.models.linear_model import LinearMMM, RidgeMMM
from mmm.models.random_forest_model import RandomForestMMM


class ModelFactory:
    @staticmethod
    def get_model(model_type: str, **kwargs):
        if model_type == "linear":
            return LinearMMM()
        elif model_type == "ridge":
            return RidgeMMM(alpha=kwargs.get("alpha", 1.0))
        elif model_type == "rf":
            return RandomForestMMM()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
