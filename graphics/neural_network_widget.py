import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer, pyqtSlot

from graphics.neural_network_chart_widget import NeuralNetworkChartWidget
from graphics.neural_network_teaching_controller_widget import NeuralNetworkTeachingControllerWidget
from initializers import RangeInitializer, UniformInitializer, ConstInitializer
from store import Store

__all__ = ['NeuralNetworkWidget']

CHART_UPDATE_INTERVAL = 200
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


class NeuralNetworkWidget(QWidget):
    def __init__(self,
                 function_code,
                 model,
                 teacher,
                 gradient,
                 error,
                 learning_rate,
                 parent=None):
        super(NeuralNetworkWidget, self).__init__(parent=parent)
        self._function_code = function_code
        self._model = model
        self._teacher = teacher
        self._gradient = gradient
        self._error = error
        self._learning_rate = learning_rate

        self._create_function()

        original_inputs = range_initializer((ORIGINAL_POINTS_COUNT, 1))
        original_values = np.concatenate((original_inputs, self._function(original_inputs)), axis=1)
        self._original_store = Store(original_values)

        self._reset_train_data_store()
        self._init_ui()

        self._timer = QTimer(self)
        self._timer.setInterval(CHART_UPDATE_INTERVAL)
        self._timer.timeout.connect(self.update_network_chart)
        self._timer.start()

    def _create_function(self):
        def function(x):
            try:
                return eval(self._function_code)
            except:
                return 0. * x
        self._function = function

    def _reset_train_data_store(self):
        original_values = self._original_store.next(TRAIN_POINTS_COUNT)
        train_values = noise(original_values)
        self._train_data_store = Store(train_values)

    def _init_ui(self):
        self._chart_widget = NeuralNetworkChartWidget(self._function_code)
        self._model_teacher_widget = NeuralNetworkTeachingControllerWidget(
            self._model,
            self._teacher,
            self._gradient,
            self._error,
            self._train_data_store,
            self._learning_rate,
            parent=self)
        self._model_teacher_widget.on_change_model = self._on_update_model

        self._neural_network_layout = QVBoxLayout(self)
        self._neural_network_layout.addWidget(self._chart_widget)
        self._neural_network_layout.addWidget(self._model_teacher_widget)

        self._init_charts()

    def _on_update_model(self, model):
        self._model = model

    def _init_charts(self):
        self._chart_widget.original_function = self._function
        self._chart_widget.train_points = self._train_data_store.values
        self._chart_widget.network_model = self._model

    @pyqtSlot()
    def update_network_chart(self):
        self._chart_widget.network_model = self._model
