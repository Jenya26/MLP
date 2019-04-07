from assertion import ok, fail, equals
from initializers import RangeInitializer

__all__ = [
    'test_range_initializer',
    'test_2d_range_initializer',
    'test_invalid_range'
]


def test_range_initializer():
    initializer = RangeInitializer()
    expected = [-1., 0.,  1.]
    equals(expected, initializer(3))


def test_2d_range_initializer():
    initializer = RangeInitializer(-3., 3.)
    expected = [
        [-3.],
        [-2.],
        [-1.],
        [0.],
        [1.],
        [2.],
        [3.]
    ]
    equals(expected, initializer((7, 1)))


def test_invalid_range():
    try:
        RangeInitializer(5., -5.)
        fail('Start can\'t be more than stop')
    except ValueError:
        ok()
