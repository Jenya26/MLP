import numpy as np


class ConstInitializer:
    def __init__(self, value=0.):
        self._value = value

    def __call__(self, shape):
        return self._value * np.ones(shape)
