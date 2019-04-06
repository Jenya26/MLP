import numpy as np


class ConstShapeInitializer:
    def __init__(self, value=np.zeros(1)):
        self._value = value

    @staticmethod
    def _normalize_shape(shape):
        if isinstance(shape, int):
            return shape,
        return shape

    def __call__(self, shape):
        shape = ConstShapeInitializer._normalize_shape(shape)
        value_shape = np.shape(self._value)
        if shape != value_shape:
            raise ValueError(f'Shape should be equals {self._value.shape}')
        return np.asarray(self._value)
