import numpy as np


class RangeInitializer:
    def __init__(self, start=-1., stop=1.):
        if start > stop:
            raise ValueError('Start should be less than stop')
        self._start = start
        self._stop = stop

    def _get_range(self, shape):
        if isinstance(shape, tuple):
            result = []
            if len(shape) == 1:
                return self._get_range(shape[0])
            count = shape[0]
            residual_shape = shape[1:]
            for i in range(count):
                result += [self._get_range(residual_shape)]
            return np.asarray(result)
        return np.linspace(self._start, self._stop, shape)

    @staticmethod
    def _get_count(shape):
        if isinstance(shape, int):
            return shape
        result = 1
        for dim in shape:
            result *= dim
        return result

    def __call__(self, shape):
        count = self._get_count(shape)
        result = np.linspace(self._start, self._stop, count)
        return result.reshape(shape)
