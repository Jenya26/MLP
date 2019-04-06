
__all__ = ['SigmoidFunction']

import numpy as np


class SigmoidFunction:
    def __call__(self, inputs):
        # fix overflow
        inputs = np.clip(inputs, -500, 500)
        return 1. / (1. + np.exp(-inputs))

    def derivative(self, inputs, outputs=None):
        if outputs is None:
            outputs = self(inputs)
        return outputs * (1 - outputs)

    def __repr__(self):
        return "SigmoidLayer()"
