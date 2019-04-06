import numpy as np

from assertion import equals
from functions import LinearFunction

__all__ = ['should_calculate_single_input', 'should_calculate_multiple_input']


def should_calculate_single_input():
    linear = LinearFunction()
    equals(12., linear(12.))


def should_calculate_multiple_input():
    linear = LinearFunction()
    equals([12., 8.], linear(np.asarray([12., 8.])))
