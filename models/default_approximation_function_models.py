import numpy as np

from functions import LinearFunction
from mlp import NeuralNetwork, Layer
from initializers import RangeInitializer, UniformInitializer
from store import Store
from gradients import Gradient
from errors import SquareError
from teacher import GradientTeacher
from .approximation_function_model import ApproximationFunctionModel

__all__ = ['models']

ORIGINAL_POINTS_COUNT = 1000
TRAIN_POINTS_COUNT = 10

range_initializer = RangeInitializer(-10., 10.)
uniform_initializer = UniformInitializer()


def noise(values):
    return values + uniform_initializer(values.shape)


def create_model(function):
    original_inputs = range_initializer((ORIGINAL_POINTS_COUNT, 1))
    original_store = Store(np.concatenate((original_inputs, function(original_inputs)), axis=1))
    return ApproximationFunctionModel(
        function=lambda x: 2 * x,
        original_store=original_store,
        train_store=Store(noise(original_store.next(100))),
        gradient=Gradient(),
        error=SquareError(),
        teacher=GradientTeacher(),
        model=NeuralNetwork([
            Layer(
                input_dimension=1,
                output_dimension=1,
                activation_function=LinearFunction()
            ),
            Layer(
                input_dimension=1,
                output_dimension=1,
                activation_function=LinearFunction(2.)
            )
        ])
    )


models = [
    create_model(lambda x: 2 * x)
]
