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

    @property
    def layers(self):
        return self._layers

    def __call__(self, inputs):
        outputs = inputs
        for layer in self._layers:
            outputs = layer(outputs)
        return outputs

    def update(self, delta):
        count = len(self.layers)
        if delta.shape[0] != count:
            raise ValueError('Count of deltas not equals count of layers')
        for i, layer in enumerate(self.layers):
            layer.update(delta[i])

    def copy(self):
        return MultipleLayersModel([layer.copy() for layer in self._layers])

    def add_layer(self):
        last_layer = self.layers[-1]
        self._layers += [
            Layer(
                 last_layer.output_dimension,
                 last_layer.output_dimension
            )
        ]

    def remove_layer(self, index):
        del self._layers[index]
        input_dimension = 1
        if 0 < index:
            layer = self._layers[index - 1]
            input_dimension = layer.output_dimension
        if index < len(self._layers):
            self._layers[index].input_dimentsion = input_dimension

    def reset(self):
        for layer in self._layers:
            layer.reset()
