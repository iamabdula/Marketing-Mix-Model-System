from mmm.service import MMMService


def get_service() -> MMMService:
    """
    Dependency injector for MMMService.
    Allows easy swapping of model type or service logic in future.
    """
    return MMMService(model_type="linear")
