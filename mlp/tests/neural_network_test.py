import numpy as np

from assertion import equals
from functions import LinearFunction
from initializers import ConstShapeInitializer
from mlp import MultipleLayersModel, Layer

__all__ = ['should_success_calculate']


def should_success_calculate():
    network = MultipleLayersModel([
        Layer(
            input_dimension=1,
            output_dimension=2,
            activation_function=LinearFunction(),
            weights_initializer=ConstShapeInitializer(
                np.asarray([
                    [1., 3.]
                ])
            ),
            biases_initializer=ConstShapeInitializer(
                np.asarray(
                    [4., 3.]
                )
            )
        ),
        Layer(
            input_dimension=2,
            output_dimension=1,
            activation_function=LinearFunction(),
            weights_initializer=ConstShapeInitializer(
                np.asarray([
                    [1.],
                    [2.]
                ])
            ),
            biases_initializer=ConstShapeInitializer(
                np.asarray(
                    [3.]
                )
            )
        )
    ])
    expected = np.asarray(
        [20.]
    )
    equals(expected, network([1.0]))
