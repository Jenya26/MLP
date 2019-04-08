from .default_approximation_function_models import models as default_models

__all__ = ['NetworkModel']


class NetworkModel:
    def __init__(self):
        self._models = default_models
        self._current_model_index = 0
        self._on_change_current_model_subscriptions = []

    @property
    def models(self):
        return self._models

    @property
    def current_model(self):
        return self._models[self._current_model_index]

    @property
    def current_model_index(self):
        return self._current_model_index

    @current_model_index.setter
    def current_model_index(self, current_model_index):
        self._current_model_index = current_model_index
        for callback in self._on_change_current_model_subscriptions:
            callback(self.current_model)

    def subscribe_on_change_current_model(self, callback):
        if not callable(callback):
            raise ValueError('Callback should be callable')
        self._on_change_current_model_subscriptions += [callback]
