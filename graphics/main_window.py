import sys

import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor


from models import NetworkModel
from graphics.chart_widget import ChartWidget
from graphics.model_teacher_controller import ModelTeachingController

__all__ = ['NeuralNetworkWindow']

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE = "Function approximation via neural network"
CHART_TITLE = "Neural network"


class NeuralNetworkWindow(QMainWindow):
    def __init__(self, network_model, parent=None):
        super(NeuralNetworkWindow, self).__init__(parent=parent)
        self._current_model = len(network_model.models) - 1
        self._network_model = network_model
        self._chart_widget = ChartWidget(CHART_TITLE)
        self.setCentralWidget(self._chart_widget)
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
        self._model_teaching_controller = ModelTeachingController(self, network_model)
        self._model_teaching_controller.start()

    @property
    def current_model(self):
        return self._network_model.models[self._current_model]

    def _get_values(self):
        current_model = self.current_model
        network = current_model.last_model
        original_values = current_model.original.values
        x_original_values = original_values[:, 0].reshape((original_values.shape[0], 1))
        y_network_values = network(x_original_values)
        train_values = current_model.train.values
        network_values = np.concatenate((x_original_values, y_network_values), axis=1)
        return original_values, train_values, network_values

    def update_network_chart(self):
        original_values, train_values, network_values = self._get_values()
        self._chart_widget.update_series(self._network_line_series, network_values)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = NeuralNetworkWindow(NetworkModel())
    window.setWindowTitle(WINDOW_TITLE)
    window.show()
    window.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

    sys.exit(app.exec_())
