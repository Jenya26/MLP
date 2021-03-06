import numpy as np

__all__ = ['GradientTeacher']


class GradientTeacher:
    def __call__(self, model, gradient, error, data_store,
                 max_iterations=1, batch=1, learning_rate=1e-3, finish_criteria=None):
        for iteration in range(max_iterations):
            data = data_store.next(batch)
            inputs, outputs = np.split(data, 2, axis=1)
            network_gradient = gradient(model, inputs, outputs, error)
            network_gradient /= batch
            network_gradient *= -learning_rate
            model.update(network_gradient)
            if finish_criteria is not None and finish_criteria(model):
                break
