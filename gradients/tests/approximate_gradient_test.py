import numpy as np

from assertion import equals
from functions import LinearFunction
from initializers import ConstShapeInitializer
from mlp import MultipleLayersModel, Layer
from errors import SquareError
from gradients import ApproximateGradient

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
    gradient = ApproximateGradient()
    square_error = SquareError()
    network_gradient = gradient(network, X, Y, square_error)
    expected = np.asarray([
        [
            np.asarray([
                [192.0000359518781]
            ]),
            np.asarray(
                [336.0000719681011]
            )
        ],
        [
            np.asarray([
                [288.0000519667192]
            ]),
            np.asarray(
                [112.00000793110121]
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
    gradient = ApproximateGradient()
    square_error = SquareError()
    network_gradient = gradient(network, X, Y, square_error)
    expected = np.asarray([
        [
            np.asarray([
                [224.00000444, 448.0000166 , 672.00003605]
            ]),
            np.asarray(
                [344.00000857,  688.0000326 , 1032.00007197]
            )
        ],
        [
            np.asarray([
                [568.00002073],
                [1136.00008012],
                [1704.00017987]
            ]),
            np.asarray(
                [344.00000834]
            )
        ]
    ])
    equals(expected, network_gradient)
