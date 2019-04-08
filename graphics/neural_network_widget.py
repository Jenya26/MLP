import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from graphics.chart_widget import ChartWidget
from graphics.model_teacher_widget import ModelTeacherWidget

__all__ = ['NeuralNetworkWidget']

CHART_TITLE = "Neural network"

class NeuralNetworkWidget(QWidget):
    def __init__(self, network_model, parent=None):
        super(NeuralNetworkWidget, self).__init__(parent=parent)
        self._current_model = len(network_model.models) - 1
        self._network_model = network_model
        self._chart_widget = ChartWidget(CHART_TITLE)
        self._model_teacher_widget = ModelTeacherWidget(network_model, self)
        self._neural_network_layout = QVBoxLayout(self)
        self._neural_network_layout.addWidget(self._chart_widget)
        self._neural_network_layout.addWidget(self._model_teacher_widget)
        self._init_charts()

    @property
    def current_model(self):
        return self._network_model.models[self._current_model]

    @property
    def current_mode_index(self):
        return self._current_model

    def change_current_model(self, index):
        if index < 0 or index >= len(self._network_model.models):
            raise ValueError('Index out of range')
        self._current_model = index
        self.update_all_charts()

    def _get_values(self):
        current_model = self.current_model
        network = current_model.last_model
        original_values = current_model.original.values
        x_original_values = original_values[:, 0].reshape((original_values.shape[0], 1))
        y_network_values = network(x_original_values)
        train_values = current_model.train.values
        network_values = np.concatenate((x_original_values, y_network_values), axis=1)
        return original_values, train_values, network_values

    def _init_charts(self):
        original_values, train_values, network_values = self._get_values()
        self._original_line_series = self._chart_widget.create_line_series(
            original_values, color=Qt.green
        )
        self._train_scatter_series = self._chart_widget.create_scatter_series(
            train_values, color=QColor(255, 0, 0)
        )
        self._network_line_series = self._chart_widget.create_line_series(
            network_values, color=QColor(255, 165, 0)
        )
        self._chart_widget.update_axes(self._original_line_series, original_values)

    def update_all_charts(self):
        original_values, train_values, network_values = self._get_values()
        self._chart_widget.update_series(self._original_line_series, original_values)
        self._chart_widget.update_series(self._train_scatter_series, train_values)
        self._chart_widget.update_series(self._network_line_series, network_values)
        self._chart_widget.update_axes(self._original_line_series, original_values)

    def update_network_chart(self):
        original_values, train_values, network_values = self._get_values()
        self._chart_widget.update_series(self._network_line_series, network_values)
