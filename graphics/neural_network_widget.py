import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer, pyqtSlot

from graphics.neural_network_chart_widget import NeuralNetworkChartWidget
from graphics.neural_network_teaching_controller_widget import NeuralNetworkTeachingControllerWidget

__all__ = ['NeuralNetworkWidget']

CHART_TITLE = "Neural network"
CHART_UPDATE_INTERVAL = 200


class NeuralNetworkWidget(QWidget):
    def __init__(self,
                 function,
                 model,
                 teacher,
                 gradient,
                 error,
                 train_data_store,
                 parent=None):
        super(NeuralNetworkWidget, self).__init__(parent=parent)
        self._function = function
        self._model = model
        self._teacher = teacher
        self._gradient = gradient
        self._error = error
        self._train_data_store = train_data_store
        self._init_ui()

        self._timer = QTimer(self)
        self._timer.setInterval(CHART_UPDATE_INTERVAL)
        self._timer.timeout.connect(self.update_network_chart)
        self._timer.start()

    def _init_ui(self):
        self._chart_widget = NeuralNetworkChartWidget(CHART_TITLE)
        self._model_teacher_widget = NeuralNetworkTeachingControllerWidget(
             self._model,
             self._teacher,
             self._gradient,
             self._error,
             self._train_data_store,
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
