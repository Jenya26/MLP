from assertion import equals
from functions import SigmoidFunction
from initializers import ConstShapeInitializer
from mlp import Layer

import numpy as np

__all__ = ['should_be_success_calculate_output']


def should_be_success_calculate_output():
    layer = Layer(
        input_dimension=2,
        output_dimension=3,
        weights_initializer=ConstShapeInitializer(
            np.asarray([
                [1., 2., 3.],
                [1., 2., 3.]
            ])
        ),
        biases_initializer=ConstShapeInitializer(
            np.asarray(
                [1., 2., 3.]
            )
        )
    )
    sigmoid = SigmoidFunction()
    expected = sigmoid(np.asarray(
        [4., 8, 12.]
    ))
    equals(expected, layer([1, 2]))
