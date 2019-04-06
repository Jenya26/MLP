import numpy as np

from assertion import ok, fail, equals
from initializers import ConstShapeInitializer

__all__ = [
    'should_be_raise_value_error',
    'should_be_return_default_array',
    'should_be_return_specific_array'
]


def should_be_return_default_array():
    initializer = ConstShapeInitializer()
    expected = [0.]
    equals(expected, initializer(1))


def should_be_raise_value_error():
    initializer = ConstShapeInitializer()
    try:
        initializer(2)
        fail('Should be raise ValueError because shape not equals')
    except ValueError:
        ok()


def should_be_return_specific_array():
    expected = np.asarray([[1., 2., 3.], [4., 5., 6.]])
    initializer = ConstShapeInitializer(expected)
    try:
        equals(expected, initializer((2, 3)))
    except ValueError:
        ok()
