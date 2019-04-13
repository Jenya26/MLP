import numpy as np

from assertion import equals
from functions import LinearFunction
from initializers import ConstShapeInitializer
from mlp import MultipleLayersModel, Layer
from errors import SquareError
from gradients import Gradient

__all__ = [
    'should_success_calculate_for_multiple_examples',
    'should_success_calculate_for_multiple_neurons'
]


def should_success_calculate_for_multiple_examples():
    network = MultipleLayersModel([
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
    X = np.asarray([[0.],
                   [1.]])
    Y = np.asarray([[0.],
                   [2.]])
    gradient = Gradient()
    square_error = SquareError()
    network_gradient = gradient(network, X, Y, square_error(Y, network(X), 1))
    expected = np.asarray([
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
    ])
    equals(expected, network_gradient)


def should_success_calculate_for_multiple_neurons():
    network = MultipleLayersModel([
        Layer(
            input_dimension=1,
            output_dimension=3,
            activation_function=LinearFunction(),
            weights_initializer=ConstShapeInitializer(
                np.asarray([
                    [1., 2., 3.]
                ])
            ),
            biases_initializer=ConstShapeInitializer(
                np.asarray(
                    [1., 2., 3.]
                )
            )
        ),
        Layer(
            input_dimension=3,
            output_dimension=1,
            activation_function=LinearFunction(2.),
            weights_initializer=ConstShapeInitializer(
                np.asarray([
                    [1.],
                    [2.],
                    [3.]
                ])
            ),
            biases_initializer=ConstShapeInitializer(
                np.asarray(
                    [1.]
                )
            )
        )
    ])
    X = np.asarray([[0.],
                   [1.]])
    Y = np.asarray([[0.],
                   [2.]])
    gradient = Gradient()
    square_error = SquareError()
    network_gradient = gradient(network, X, Y, square_error(Y, network(X), 1))
    expected = np.asarray([
        [
            np.asarray([
                [224., 448., 672.]
            ]),
            np.asarray(
                [344.,  688., 1032.]
            )
        ],
        [
            np.asarray([
                [568.],
                [1136.],
                [1704.]
            ]),
            np.asarray(
                [344.]
            )
        ]
    ])
    equals(expected, network_gradient)
