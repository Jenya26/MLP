__all__ = ['SquareError']


class SquareError:
    def __call__(self, expected, actual, derivative=0):
        if derivative < 0:
            raise ValueError('Derivative can\'t be zero')
        if derivative > 2:
            return 0.
        if derivative == 2:
            return 2.
        if derivative == 1:
            return 2 * (actual - expected)
        return (actual - expected) ** 2
