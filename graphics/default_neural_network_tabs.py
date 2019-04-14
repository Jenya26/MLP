from functions import LinearFunction
from mlp import MultipleLayersModel, Layer
from gradients import Gradient
from errors import SquareError
from teacher import GradientTeacher

__all = ['tabs']


def create_tab(function_text, model, learning_rate=1e-3):
    return {
        'function': function_text,
        'model': model,
        'gradient': Gradient(),
        'error': SquareError(),
        'teacher': GradientTeacher(),
        'learning_rate': learning_rate
    }


tabs = [
    create_tab(
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
    create_tab(
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
    create_tab(
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
    create_tab(
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
    create_tab(
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
    create_tab(
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
    create_tab(
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
