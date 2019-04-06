
__all__ = ['LinearFunction']


class LinearFunction:
    def __init__(self, weight=1., bias=0.):
        self._weight = weight
        self._bias = bias

    def __call__(self, inputs):
        return self._weight * inputs + self._bias

    def derivative(self, inputs, outputs=None):
        return self._weight

    def __repr__(self):
        return "LinearFunction()"
