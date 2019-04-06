import numpy as np

from assertion import equals
from functions import LinearFunction
from initializers import ConstShapeInitializer
from mlp import NeuralNetwork, Layer
from errors import SquareError
from gradients import Gradient

__all__ = ['should_success_calculate_for_multiple_examples']

network = NeuralNetwork([
    Layer(
        input_dimension=1,
        output_dimension=1,
        activation_function=LinearFunction(),
        weights_initializer=ConstShapeInitializer(
            np.asarray([
                [1.]
            ])
        ),
        biases_initializer=ConstShapeInitializer(
            np.asarray(
                [2.]
            )
        )
    ),
    Layer(
        input_dimension=1,
        output_dimension=1,
        activation_function=LinearFunction(2.),
        weights_initializer=ConstShapeInitializer(
            np.asarray([
                [3.]
            ])
        ),
        biases_initializer=ConstShapeInitializer(
            np.asarray(
                [0.]
            )
        )
    )
])


def should_success_calculate_for_multiple_examples():
    X = np.asarray([[0.],
                   [1.]])
    Y = np.asarray([[0.],
                   [2.]])
    gradient = Gradient()
    square_error = SquareError()
    network_gradient = gradient(network, X, Y, square_error(Y, network(X), 1))
    expected = [
        [
            np.asarray([
                [192.]
            ]),
            np.asarray(
                [336.]
            )
        ],
        [
            np.asarray([
                [288.]
            ]),
            np.asarray(
                [112.]
            )
        ]
    ]
    equals(expected, network_gradient)
