__all__ = ['NeuralNetwork']

class NeuralNetwork:
    def __init__(self, layers):
        for i, layer in enumerate(layers[1:]):
            input_dimension = layer.input_dimension
            output_dimension = layers[i].output_dimension
            if output_dimension != input_dimension:
                raise ValueError(
                    f'Output dimension({output_dimension}) not equals input dimension({input_dimension})'
                )
            self._layers = layers

    def __call__(self, inputs):
        outputs = inputs
        for layer in self._layers:
            outputs = layer(outputs)
        return outputs
