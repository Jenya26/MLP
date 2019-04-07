import numpy as np

from initializers import UniformInitializer
from functions import SigmoidFunction

__all__ = ['Layer']


class Layer:
    def __init__(self,
                 input_dimension,
                 output_dimension,
                 activation_function=SigmoidFunction(),
                 weights_initializer=UniformInitializer(),
                 biases_initializer=UniformInitializer()
                 ):
        self._input_dimension = input_dimension
        self._output_dimension = output_dimension
        self._activation_function = activation_function
        self._weights = weights_initializer((input_dimension, output_dimension))
        self._biases = biases_initializer(output_dimension)

    @property
    def input_dimension(self):
        return self._input_dimension

    @property
    def output_dimension(self):
        return self._output_dimension

    @property
    def activation_function(self):
        return self._activation_function

    @property
    def weights(self):
        return self._weights

    @staticmethod
    def _is_number(value):
        """
        Return true if value is number

        """
        if isinstance(value, int) or isinstance(value, float):
            return True
        return False

    @staticmethod
    def _normalize_inputs(inputs):
        """
        Return normalized inputs
        ==========================================================

        transform:
        - vector of shape (k, n) to vector of shape (k, n)
        - number to vector of shape (1, 1) if input dimension equals 1

        """
        if Layer._is_number(inputs):
            return np.asarray([inputs])
        if isinstance(inputs, list):
            return np.asarray(inputs)
        return inputs

    def __call__(self, inputs):
        """
        Transform n dimensional vector to m dimensional vector
        using affine transformation and apply activation functions
        for output m dimensional vector
        ==========================================================

        k - count of inputs

        inputs - vector of shape (k, n)
        weights - matrix of shape (n, m)
        biases - vector of shape (1, m)

        """
        inputs = Layer._normalize_inputs(inputs)
        outputs = inputs.dot(self._weights) + self._biases
        return self._activation_function(outputs)

    def update(self, delta):
        if delta.shape[0] != 2:
            raise ValueError('Count of delta should be equals two')

        weight_dtype = self._weights.dtype
        delta_weights = delta[0].reshape((self.input_dimension, self.output_dimension)).astype(weight_dtype)
        bias_type = self._biases.dtype
        delta_biases = delta[1].reshape(self.output_dimension).astype(bias_type)
        self._weights += delta_weights
        self._biases += delta_biases
