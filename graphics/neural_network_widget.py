import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer, pyqtSlot

from graphics.neural_network_chart_widget import NeuralNetworkChartWidget
from graphics.neural_network_teaching_controller_widget import NeuralNetworkTeachingControllerWidget

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
        self._chart_widget = NeuralNetworkChartWidget(CHART_TITLE)
        current_model = self._network_model.current_model
        self._model_teacher_widget = NeuralNetworkTeachingControllerWidget(
             current_model.current_model,
             current_model.teacher,
             current_model.gradient,
             current_model.error,
             current_model.train,
             parent=self)
        self._model_teacher_widget.on_change_model = self._on_update_model

        self._neural_network_layout = QVBoxLayout(self)
        self._neural_network_layout.addWidget(self._chart_widget)
        self._neural_network_layout.addWidget(self._model_teacher_widget)

        self._init_charts()

    def _on_update_model(self, model):
        current_model = self._network_model.current_model
        current_model.add_model(model)

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
        self._chart_widget.original_function = current_model.function
        self._chart_widget.train_points = train_values
        self._chart_widget.network_model = current_model.current_model

    def update_all_charts(self, current_model):
        original_values, train_values, network_values = self._get_values(
            current_model.current_model, current_model.original.values, current_model.train.values
        )
        self._chart_widget.original_function = current_model.function
        self._chart_widget.train_points = train_values
        self._chart_widget.network_model = current_model.current_model

    @pyqtSlot()
    def update_network_chart(self):
        network_model = self._network_model
        current_model = network_model.current_model
        self._chart_widget.network_model = current_model.current_model
