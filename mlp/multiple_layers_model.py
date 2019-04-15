from PyQt5.QtCore import QMutex

from .layer import Layer

__all__ = ['MultipleLayersModel']


class MultipleLayersModel:
    def __init__(self, layers):
        if len(layers) <= 0:
            raise ValueError('Layers should be more or equals one')
        for i, layer in enumerate(layers[1:]):
            input_dimension = layer.input_dimension
            output_dimension = layers[i].output_dimension
            if output_dimension != input_dimension:
                raise ValueError(
                    f'Output dimension({output_dimension}) not equals input dimension({input_dimension})'
                )
        self._layers = layers
        self._mutex = QMutex()

    @property
    def layers(self):
        self._mutex.lock()
        layers = self._layers
        self._mutex.unlock()
        return layers

    def __call__(self, inputs):
        self._mutex.lock()
        outputs = inputs
        layers = self._layers
        for layer in layers:
            outputs = layer(outputs)
        self._mutex.unlock()
        return outputs

    def update(self, delta):
        self._mutex.lock()
        count = len(self._layers)
        if delta.shape[0] != count:
            raise ValueError('Count of deltas not equals count of layers')
        for i, layer in enumerate(self._layers):
            layer.update(delta[i])
        self._mutex.unlock()

    def copy(self):
        self._mutex.lock()
        copy = MultipleLayersModel([layer.copy() for layer in self._layers])
        self._mutex.unlock()
        return copy

    def add_layer(self):
        self._mutex.lock()
        last_layer = self._layers[-1]
        self._layers += [
            Layer(
                 last_layer.output_dimension,
                 last_layer.output_dimension
            )
        ]
        self._mutex.unlock()

    def remove_layer(self, index):
        self._mutex.lock()
        del self._layers[index]
        input_dimension = 1
        if 0 < index:
            layer = self._layers[index - 1]
            input_dimension = layer.output_dimension
        if index < len(self._layers):
            self._layers[index].input_dimension = input_dimension
        self._mutex.unlock()

    def reset(self):
        self._mutex.lock()
        for layer in self._layers:
            layer.reset()
        self._mutex.unlock()

    def change_layer_count(self, index, layer_count):
        self._mutex.lock()
        layers = self._layers
        if index < len(layers):
            layers[index].output_dimension = layer_count
        if index + 1 < len(layers):
            layers[index + 1].input_dimension = layer_count
        self._mutex.unlock()

    def change_activation_function(self, index, function):
        self._mutex.lock()
        self._layers[index].activation_function = function
        self._mutex.unlock()

    def lock_layers(self):
        self._mutex.lock()
        return self._layers

    def unlock(self):
        self._mutex.unlock()
