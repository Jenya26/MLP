import numpy as np


class UniformInitializer:
    def __init__(self,
                 low=-1.,
                 high=1.,
                 seed=None
                 ):
        self._low = low
        self._high = high
        self._random = np.random.RandomState(seed)

    def __call__(self, shape):
        return self._random.uniform(self._low, self._high, shape)
