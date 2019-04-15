import numpy as np

__all__ = ['Gradient']


class Gradient:
    @staticmethod
    def _forward(network, x, y):
        layer_inputs = [x]
        layer_outputs = [network.layers[0](x)]
        for i, layer in enumerate(network.layers[1:]):
            layer_inputs += [layer_outputs[i]]
            layer_outputs += [layer(layer_inputs[i + 1])]
        return layer_inputs, layer_outputs

    @staticmethod
    def _normalize_vectors(vectors):
        if isinstance(vectors, float):
            return np.asarray([vectors])
        return vectors

    @staticmethod
    def _calculate_layer_gradient(gradient, layer, inputs, outputs, outputs_gradient):
        activation_function_gradient = outputs_gradient * layer.activation_function.derivative(inputs, outputs)
        weights_gradient = np.zeros((layer.input_dimension, layer.output_dimension))
        for i in range(inputs.shape[0]):
            current_input = inputs[i].reshape((layer.input_dimension, 1))
            current_output = activation_function_gradient[i].reshape((1, layer.output_dimension))
            weights_gradient += np.dot(current_input, current_output)
        biases_gradient = np.sum(activation_function_gradient, axis=0)
        gradient += [[
            weights_gradient,
            biases_gradient
        ]]
        return activation_function_gradient.dot(layer.weights.T)

    def __call__(self, network, inputs, outputs, error):
        inputs = Gradient._normalize_vectors(inputs)
        outputs = Gradient._normalize_vectors(outputs)
        layer_inputs, layer_outputs = Gradient._forward(network, inputs, outputs)
        outputs_gradient = error(outputs, layer_outputs[-1], 1)

        gradient = []
        for i, layer in enumerate(network.layers[::-1]):
            outputs_gradient = Gradient._calculate_layer_gradient(
                gradient,
                layer,
                layer_inputs[-1 - i],
                layer_outputs[-1 - i],
                outputs_gradient
            )
        return np.asarray(gradient[::-1])
