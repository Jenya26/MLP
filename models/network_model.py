from .default_approximation_function_models import models as default_models

__all__ = ['NetworkModel']


class NetworkModel:
    def __init__(self):
        self._models = default_models

    @property
    def models(self):
        return self._models
