from mmm.models.linear_model import LinearMMM, RidgeMMM


class ModelFactory:
    """
    Factory class to create model instances dynamically.
    """

    @staticmethod
    def get_model(model_type: str, **kwargs):
        if model_type == "linear":
            return LinearMMM()
        elif model_type == "ridge":
            alpha = kwargs.get("alpha", 1.0)
            return RidgeMMM(alpha=alpha)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
