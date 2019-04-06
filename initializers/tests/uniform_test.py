from assertion import equals
from initializers import UniformInitializer

__all__ = ['test_uniform_initializer']


def test_uniform_initializer():
    initializer = UniformInitializer(
        seed=2019
    )
    expected = [0.80696443, -0.21383899,  0.24793992,  0.2757548]
    equals(expected, initializer(4))
