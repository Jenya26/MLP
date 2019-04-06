import numpy as np

from assertion import equals
from store import Store

__all__ = [
    'success_get_all_values',
    'success_get_next_two_values',
    'success_get_batch_more_than_values_count'
]


def success_get_all_values():
    values = [
        np.asarray([1., 2., 3.]),
        np.asarray([4., 5., 6.]),
        np.asarray([7., 8., 9.])
    ]
    store = Store(values)
    equals(values, store.values)


def success_get_next_two_values():
    values = [
        np.asarray([1., 2., 3.]),
        np.asarray([4., 5., 6.]),
        np.asarray([7., 8., 9.])
    ]
    store = Store(values, 2019)
    next_values = store.next(2)
    equals(values[1:], next_values)
    next_values = store.next(1)
    equals(values[:1], next_values)


def success_get_batch_more_than_values_count():
    values = [
        np.asarray([1., 2., 3.]),
        np.asarray([4., 5., 6.]),
        np.asarray([7., 8., 9.])
    ]
    store = Store(values, 2019)
    next_values = store.next(4)
    expected = np.asarray([
        values[1],
        values[2],
        values[0],
        values[2]
    ])
    equals(expected, next_values)
