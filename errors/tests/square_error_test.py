import numpy as np

from assertion import ok, fail, equals
from errors import SquareError

__all__ = [
    'should_success_calculate_error',
    'should_success_calculate_multiple_errors',
    'should_success_calculate_first_derivative',
    'should_success_calculate_second_derivative',
    'should_success_calculate_third_derivative',
    'should_raise_value_error_for_negative_derivative_count'
]


def should_success_calculate_error():
    square_error = SquareError()
    equals(100., square_error(10., 20.))


def should_success_calculate_multiple_errors():
    Y = np.asarray([0., 2.])
    R = np.asarray([12., 18.])
    E = np.asarray([144., 256.])
    square_error = SquareError()
    equals(E, square_error(Y, R))


def should_success_calculate_first_derivative():
    square_error = SquareError()
    equals(20., square_error(10., 20., 1))


def should_success_calculate_second_derivative():
    square_error = SquareError()
    equals(2., square_error(10., 20., 2))


def should_success_calculate_third_derivative():
    square_error = SquareError()
    equals(0., square_error(10., 20., 3))


def should_raise_value_error_for_negative_derivative_count():
    square_error = SquareError()
    try:
        equals(2., square_error(10., 20., -3))
        fail('Should be raise value error because derivative count is negative')
    except ValueError:
        ok()
