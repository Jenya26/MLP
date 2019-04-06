import numpy as np

__all__ = ['GradientTeacher']


class GradientTeacher:
    def __call__(self, network, gradient, error, dataStore,
                 max_iterations=1, batch=1, learning_rate=1e-3, finish_criteria=None):
        for iteration in range(max_iterations):
            data = dataStore.next(batch)
            inputs, outputs = np.split(data, 2, axis=1)
            actual = network(inputs)
            outputs_gradient = error(outputs, actual, 1)
            network_gradient = gradient(network, inputs, outputs, outputs_gradient)
            network_gradient /= batch
            network_gradient *= -learning_rate
            network.update(network_gradient)
            if finish_criteria is not None and finish_criteria(network):
                break
