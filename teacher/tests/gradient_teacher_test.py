from functions import LinearFunction
from store import Store
from errors import SquareError
from gradients import Gradient
from teacher import GradientTeacher
from mlp import NeuralNetwork, Layer
from initializers import UniformInitializer, ConstShapeInitializer

import numpy as np

__all__=[
    'gradient_teacher_test'
]


def function(x):
    return 2 * x


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


def gradient_teacher_test():
    uniform = UniformInitializer(
        seed=2019
    )
    inputs = uniform((5, 1))
    outputInitializer = ConstShapeInitializer([function(value) for value in inputs])
    outputs = outputInitializer((5, 1))
    dataStore = Store(np.concatenate((inputs, outputs), axis=1))
    square_error = SquareError()
    gradient = Gradient()
    teacher = GradientTeacher()

    teacher(network, gradient, square_error, dataStore, 1000, 5)
