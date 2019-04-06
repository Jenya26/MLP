from assertion import equals
from initializers import ConstInitializer

__all__ = ['test_const_initializer']


def test_const_initializer():
    initializer = ConstInitializer(12.)
    expected = [12., 12.,  12.]
    equals(expected, initializer(3))
