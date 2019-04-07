__all__ = ['ApproximationFunctionModel']


class ApproximationFunctionModel:
    def __init__(self,
                 function,
                 model,
                 original_store,
                 train_store):
        self._function = function
        self._base_model = model
        self._models = [model]
        self._original_store = original_store
        self._train_store = train_store

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