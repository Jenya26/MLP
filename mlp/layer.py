import numpy as np

from initializers import UniformInitializer
from functions import LinearFunction

__all__ = ['Layer']


class Layer:
    def __init__(self,
                 input_dimension,
                 output_dimension,
                 activation_function=LinearFunction(),
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

    """
    Return true if value is number

    """
    @staticmethod
    def _is_number(value):
        if isinstance(value, int) or isinstance(value, float):
            return True
        return False
    """
    Return normalized inputs
    ==========================================================
    
    transform:
    - vector of shape (k, n) to vector of shape (k, n)
    - number to vector of shape (1, 1) if input dimension equals 1

    """
    @staticmethod
    def _normalize_inputs(inputs):
        if Layer._is_number(inputs):
            return np.asarray([inputs])
        if isinstance(inputs, list):
            return np.asarray(inputs)
        return inputs

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
    def __call__(self, inputs):
        inputs = Layer._normalize_inputs(inputs)
        outputs = inputs.dot(self._weights) + self._biases
        return self._activation_function(outputs)
