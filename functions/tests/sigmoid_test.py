from assertion import equals
from functions import SigmoidFunction

__all__ = ['sigmoid_test', 'overflow_sigmoid_test']


def sigmoid_test():
    sigmoid = SigmoidFunction()
    equals(0.5, sigmoid(0.))


def overflow_sigmoid_test():
    sigmoid = SigmoidFunction()
    equals(1., sigmoid(1000.))
