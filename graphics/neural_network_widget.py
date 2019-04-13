import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QColor

from graphics.chart_widget import ChartWidget
from graphics.model_teacher_widget import ModelTeacherWidget

__all__ = ['NeuralNetworkWidget']

CHART_TITLE = "Neural network"
CHART_UPDATE_INTERVAL = 200


class NeuralNetworkWidget(QWidget):
    def __init__(self, network_model, parent=None):
        super(NeuralNetworkWidget, self).__init__(parent=parent)
        self._network_model = network_model
        self._network_model.subscribe_on_change_current_model(self.update_all_charts)
        self._init_ui()

        self._timer = QTimer(self)
        self._timer.setInterval(CHART_UPDATE_INTERVAL)
        self._timer.timeout.connect(self.update_network_chart)
        self._timer.start()

    def _init_ui(self):
        self._chart_widget = ChartWidget(CHART_TITLE)
        self._model_teacher_widget = ModelTeacherWidget(self._network_model, self)

        self._neural_network_layout = QVBoxLayout(self)
        self._neural_network_layout.addWidget(self._chart_widget)
        self._neural_network_layout.addWidget(self._model_teacher_widget)

        self._init_charts()

    @staticmethod
    def _get_values(network, original_values, train_values):
        x_original_values = original_values[:, 0].reshape((original_values.shape[0], 1))
        y_network_values = network(x_original_values)
        network_values = np.concatenate((x_original_values, y_network_values), axis=1)
        return original_values, train_values, network_values

    def _init_charts(self):
        network_model = self._network_model
        current_model = network_model.current_model
        original_values, train_values, network_values = self._get_values(
            current_model.last_model, current_model.original.values, current_model.train.values
        )
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

    def update_all_charts(self, current_model):
        original_values, train_values, network_values = self._get_values(
            current_model.current_model, current_model.original.values, current_model.train.values
        )
        self._chart_widget.update_series(self._original_line_series, original_values)
        self._chart_widget.update_series(self._train_scatter_series, train_values)
        self._chart_widget.update_series(self._network_line_series, network_values)
        self._chart_widget.update_axes(self._original_line_series, original_values)

    @pyqtSlot()
    def update_network_chart(self):
        network_model = self._network_model
        current_model = network_model.current_model
        original_values, train_values, network_values = self._get_values(
            current_model.current_model, current_model.original.values, current_model.train.values
        )
        self._chart_widget.update_series(self._network_line_series, network_values)
