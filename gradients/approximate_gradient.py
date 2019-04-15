import numpy as np

__all__ = ['ApproximateGradient']


class ApproximateGradient:
    def __init__(self, delta=1e-6):
        self._delta = delta

    @staticmethod
    def _forward(layers, x):
        if len(layers) < 1:
            return [], []
        layer_inputs = [x]
        layer_outputs = [layers[0](x)]
        for i, layer in enumerate(layers[1:]):
            layer_inputs += [layer_outputs[i]]
            layer_outputs += [layer(layer_inputs[i + 1])]
        return layer_inputs, layer_outputs

    @staticmethod
    def _result(layers, x):
        if len(layers) < 1:
            return x
        layer_inputs, layer_outputs = ApproximateGradient._forward(layers, x)
        return layer_outputs[-1]

    @staticmethod
    def _normalize_vectors(vectors):
        if isinstance(vectors, float):
            return np.asarray([vectors])
        return vectors

    def _calculate_layer_weight_gradient(self, i, j, inputs, layer, next_layers, outputs_error, outputs, error):
        weight = layer.weights[i][j]
        layer.weights[i][j] += self._delta
        layer_outputs = layer(inputs)
        delta_model_outputs = ApproximateGradient._result(next_layers, layer_outputs)
        delta_outputs_error = error(outputs, delta_model_outputs)
        layer.weights[i][j] = weight
        delta_errors = delta_outputs_error - outputs_error
        approximate_gradient = np.sum(delta_errors / self._delta)
        return approximate_gradient

    def _calculate_layer_bias_gradient(self, i, inputs, layer, next_layers, outputs_error, outputs, error):
        bias = layer.biases[i]
        layer.biases[i] += self._delta
        delta_model_outputs = ApproximateGradient._result(next_layers, layer(inputs))
        delta_outputs_error = error(outputs, delta_model_outputs)
        layer.biases[i] = bias
        return np.sum((delta_outputs_error - outputs_error) / self._delta)

    def _calculate_layer_gradient(self, gradient, inputs, layer, next_layers, outputs_error, model_outputs, error):
        weights_gradient = np.zeros((layer.input_dimension, layer.output_dimension))
        biases_gradient = np.zeros(layer.output_dimension)
        for i in range(layer.input_dimension):
            for j in range(layer.output_dimension):
                weights_gradient[i][j] = self._calculate_layer_weight_gradient(
                    i, j, inputs, layer, next_layers, outputs_error, model_outputs, error
                )
        for i in range(layer.output_dimension):
            biases_gradient[i] = self._calculate_layer_bias_gradient(
                i, inputs, layer, next_layers, outputs_error, model_outputs, error
            )
        gradient += [[
            weights_gradient,
            biases_gradient
        ]]

    def __call__(self, network, inputs, outputs, error):
        inputs = ApproximateGradient._normalize_vectors(inputs)
        outputs = ApproximateGradient._normalize_vectors(outputs)
        layers = network.lock_layers()
        layer_inputs, layer_outputs = ApproximateGradient._forward(layers, inputs)
        model_outputs = layer_outputs[-1]
        outputs_error = error(outputs, model_outputs)

        gradient = []
        for i, layer in enumerate(layers):
            self._calculate_layer_gradient(
                gradient,
                layer_inputs[i],
                layer,
                layers[i + 1:],
                outputs_error,
                outputs,
                error
            )
        network.unlock()
        return np.asarray(gradient)
