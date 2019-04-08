from gradients import Gradient
from errors import SquareError
from teacher import GradientTeacher

__all__ = ['ApproximationFunctionModel']


class ApproximationFunctionModel:
    def __init__(self,
                 function,
                 function_text,
                 model,
                 original_store,
                 train_store,
                 gradient=Gradient(),
                 error=SquareError(),
                 teacher=GradientTeacher(),
                 learning_rate=1e-3):
        self._function_text = function_text
        self._function = function
        self._base_model = model
        self._models = [model]
        self._original_store = original_store
        self._train_store = train_store
        self._gradient = gradient
        self._error = error
        self._teacher = teacher
        self._learning_rate = learning_rate
        self._on_add_model_subscriptions = []
        self._current_model_index = 0

    @property
    def current_model(self):
        return self._models[self._current_model_index]

    @property
    def current_model_index(self):
        return self._current_model_index

    @current_model_index.setter
    def current_model_index(self, current_model_index):
        self._current_model_index = current_model_index

    def subscribe_on_add_model(self, callback):
        if not callable(callback):
            raise ValueError('Callback should be callable')
        self._on_add_model_subscriptions += [callback]

    @property
    def function_text(self):
        return self._function_text

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def gradient(self):
        return self._gradient

    @property
    def error(self):
        return self._error

    @property
    def teacher(self):
        return self._teacher

    @property
    def original(self):
        return self._original_store

    @property
    def train(self):
        return self._train_store

    @property
    def function(self):
        return self._function

    @property
    def base_model(self):
        return self._base_model

    def add_model(self, model):
        self._models += [model]
        self._current_model_index = len(self._models) - 1
        for callback in self._on_add_model_subscriptions:
            callback(model)

    def get_model(self, index):
        if 0 <= index < self.models_count:
            return self._models[index]
        raise ValueError(f'Can\'t find model with index {index}')

    @property
    def models_count(self):
        return len(self._models)

    @property
    def last_model(self):
        return self._models[self.models_count - 1]
