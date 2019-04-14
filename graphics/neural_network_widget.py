import time

import numpy as np
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer, pyqtSlot

from graphics.neural_network_chart_widget import NeuralNetworkChartWidget
from graphics.neural_network_teaching_controller_widget import NeuralNetworkTeachingControllerWidget
from graphics.neural_network_model_controller_widget import NeuralNetworkModelControllerWidget
from initializers import RangeInitializer, UniformInitializer, ConstInitializer
from store import Store

__all__ = ['NeuralNetworkWidget']

CHART_UPDATE_INTERVAL = 1
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

        self._on_change_function = None

        self._create_function()

        original_inputs = range_initializer((ORIGINAL_POINTS_COUNT, 1))
        original_values = np.concatenate((original_inputs, self._function(original_inputs)), axis=1)
        self._original_store = Store(original_values)

        self._last_update_model_time = time.time()

        self._reset_train_data_store()
        self._init_ui()

        self._timer = QTimer()
        self._timer.setInterval(CHART_UPDATE_INTERVAL)
        self._timer.timeout.connect(self.__update_model_chart)
        # self._timer.start()

        self._model_teaching_controller_widget.start(1)

    def _create_function(self):
        def function(x):
            try:
                return eval(self._function_code)
            except:
                return 0. * x
        self._function = function

    @property
    def on_change_function(self):
        return self._on_change_function

    @on_change_function.setter
    def on_change_function(self, on_change_function):
        if not callable(on_change_function):
            raise ValueError('on_change_function should be callable')
        self._on_change_function = on_change_function

    def __on_change_function(self, function_code):
        self._function_code = function_code
        self._create_function()
        self._reset_train_data_store()
        self._update_all_charts()
        if self._on_change_function is not None:
            self._on_change_function(function_code)

    def _reset_train_data_store(self):
        original_values = self._original_store.next(TRAIN_POINTS_COUNT)
        train_values = noise(original_values)
        self._train_data_store = Store(train_values)

    def _init_ui(self):
        self._chart_widget = NeuralNetworkChartWidget(self._function_code)
        self._model_teaching_controller_widget = NeuralNetworkTeachingControllerWidget(
            self._function_code,
            self._model,
            self._teacher,
            self._gradient,
            self._error,
            self._train_data_store,
            self._learning_rate,
            parent=self)
        self._model_teaching_controller_widget.on_change_model = self._on_update_model
        self._model_teaching_controller_widget.on_start_teaching = self.__on_start_teaching
        self._model_teaching_controller_widget.on_stop_teaching = self.__on_stop_teaching
        self._model_teaching_controller_widget.on_change_function = self.__on_change_function

        self._neural_network_model_controller_widget = NeuralNetworkModelControllerWidget(
            self._model
        )

        self._neural_network_layout = QVBoxLayout()
        self._neural_network_layout.addWidget(self._chart_widget, alignment=Qt.Alignment())
        self._neural_network_layout.addWidget(self._model_teaching_controller_widget, alignment=Qt.AlignBottom)

        self._neural_network_row_layout = QHBoxLayout(self)
        self._neural_network_row_layout.addWidget(self._neural_network_model_controller_widget, alignment=Qt.AlignLeft)
        self._neural_network_row_layout.addLayout(self._neural_network_layout)

        self._update_all_charts()

    def _update_model(self):
        current_time = time.time()
        delta_time = 1000. * (current_time - self._last_update_model_time)
        while delta_time <= CHART_UPDATE_INTERVAL:
            time.sleep((CHART_UPDATE_INTERVAL - delta_time) / 1000.)
            current_time = time.time()
            delta_time = 1000. * (current_time - self._last_update_model_time)
        self._last_update_model_time = current_time
        self.__update_model_chart()

    def __change_current_model(self, model):
        self._model = model
        if self._neural_network_model_controller_widget.isEnabled():
            self.__update_model_chart()
            self._neural_network_model_controller_widget.model = model
        # self._update_model()

    def _on_update_model(self, model):
        self.__change_current_model(model)

    def _update_all_charts(self):
        self._chart_widget.original_function = self._function
        self._chart_widget.train_points = self._train_data_store.values
        self._chart_widget.network_model = self._model

    def __on_start_teaching(self):
        self._timer.start()
        self._neural_network_model_controller_widget.setEnabled(False)

    def __on_stop_teaching(self, iterations, model):
        self._timer.stop()
        self._neural_network_model_controller_widget.setEnabled(True)

    @pyqtSlot(name="Update chart")
    def __update_model_chart(self):
        self._chart_widget.network_model = self._model
