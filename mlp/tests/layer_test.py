from assertion import equals
from functions import LinearFunction
from initializers import ConstShapeInitializer
from mlp import Layer

import numpy as np

__all__ = ['should_be_success_calculate_output']


def should_be_success_calculate_output():
    layer = Layer(
        input_dimension=2,
        output_dimension=3,
        activation_function=LinearFunction(),
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
    expected = np.asarray(
        [4., 8, 12.]
    )
    equals(expected, layer([1, 2]))
