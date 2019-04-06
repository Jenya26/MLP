import numpy as np

__all__ = ['Store']


class Store:
    def __init__(self, values, seed=None):
        if isinstance(values, list):
            values = np.asarray(values)
        self._values = values
        self._random = np.random.RandomState(seed)
        self._order = self._random.permutation(values.shape[0])
        self._last_order_index = 0

    @property
    def values(self):
        return self._values

    def _change_order(self):
        self._order = self._random.permutation(self._values.shape[0])
        self._last_order_index = 0

    def _next(self, batch):
        if batch <= 0:
            raise ValueError('Batch size can\'t be less than one')

        result = []
        left = self._last_order_index
        right = self._last_order_index + batch
        max_right = self._values.shape[0]
        while right > max_right:
            residual = max_right - left
            result += self._next(residual)
            left = 0
            right -= residual
            batch -= residual
        for index in self._order[left:right]:
            result += [self.values[index]]
        self._last_order_index = right
        if self._last_order_index == max_right:
            self._change_order()
        return result

    def next(self, batch):
        return np.asarray(self._next(batch))
