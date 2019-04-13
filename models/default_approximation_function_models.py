import numpy as np

from functions import LinearFunction
from mlp import MultipleLayersModel, Layer
from initializers import RangeInitializer, UniformInitializer, ConstInitializer
from store import Store
from .approximation_function_model import ApproximationFunctionModel

__all__ = ['models']

ORIGINAL_POINTS_COUNT = 1000
TRAIN_POINTS_COUNT = 10

zero_initializer = ConstInitializer(0.)
range_initializer = RangeInitializer(-2., 2.)
uniform_initializer = UniformInitializer(-.1, .1)


def noise(values):
    x_delta = zero_initializer((values.shape[0], 1))
    y_delta = uniform_initializer((values.shape[0], 1))
    delta = np.concatenate((x_delta, y_delta), axis=1)
    return values + delta


def create_model(function_text, model, learning_rate=1e-3):
    def function(x):
        return eval(function_text)
    original_inputs = range_initializer((ORIGINAL_POINTS_COUNT, 1))
    original_values = np.concatenate((original_inputs, function(original_inputs)), axis=1)
    original_store = Store(original_values)
    return ApproximationFunctionModel(
        function=function,
        function_text=function_text,
        original_store=original_store,
        learning_rate=learning_rate,
        train_store=Store(noise(original_store.next(TRAIN_POINTS_COUNT))),
        model=model
    )


models = [
    create_model(
        function_text="2 * x",
        model=MultipleLayersModel([
            Layer(
                input_dimension=1,
                output_dimension=1,
                activation_function=LinearFunction()
            ),
            Layer(
                input_dimension=1,
                output_dimension=1,
                activation_function=LinearFunction()
            )
        ])
    ),
    create_model(
        function_text="50 * x",
        learning_rate=1e-4,
        model=MultipleLayersModel([
            Layer(
                input_dimension=1,
                output_dimension=1,
                activation_function=LinearFunction()
            ),
            Layer(
                input_dimension=1,
                output_dimension=1,
                activation_function=LinearFunction()
            )
        ])
    ),
    create_model(
        function_text="x ** 2",
        learning_rate=1e-3,
        model=MultipleLayersModel([
            Layer(1, 3),
            Layer(
                input_dimension=3,
                output_dimension=1,
                activation_function=LinearFunction()
            )
        ]),
    ),
    create_model(
        function_text="np.cos(2 * np.pi * x)",
        learning_rate=1e-1,
        model=MultipleLayersModel([
            Layer(1, 5),
            Layer(5, 5),
            Layer(
                input_dimension=5,
                output_dimension=1,
                activation_function=LinearFunction()
            )
        ])
    ),
    create_model(
        function_text="x * np.sin(2. * np.pi * x)",
        learning_rate=1e-1,
        model=MultipleLayersModel([
            Layer(1, 5),
            Layer(5, 5),
            Layer(
                input_dimension=5,
                output_dimension=1,
                activation_function=LinearFunction()
            )
        ])
    ),
    create_model(
        function_text="5 * (x ** 3) + (x ** 2) + 5",
        model=MultipleLayersModel([
            Layer(1, 5),
            Layer(5, 5),
            Layer(
                input_dimension=5,
                output_dimension=1,
                activation_function=LinearFunction()
            )
        ]),
        learning_rate=1e-3
    ),
    create_model(
        function_text="5 * (x ** 7) + (x ** 2) + 5",
        model=MultipleLayersModel([
            Layer(1, 5),
            Layer(5, 5),
            Layer(
                input_dimension=5,
                output_dimension=1,
                activation_function=LinearFunction()
            )
        ]),
        learning_rate=1e-4
    )
]
