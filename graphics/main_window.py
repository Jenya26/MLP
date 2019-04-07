import sys

import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor


from models import NetworkModel
from graphics.chart_widget import ChartWidget

__all__ = ['NeuralNetworkWindow']

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE = "Function approximation via neural network"
CHART_TITLE = "Neural network"


class NeuralNetworkWindow(QMainWindow):
    def __init__(self, network_model, parent=None):
        super(NeuralNetworkWindow, self).__init__(parent=parent)
        self._chart_widget = ChartWidget(CHART_TITLE)
        network = network_model.models[0].base_model
        originalValues = network_model.models[0].original.values
        xOriginalValues = originalValues[:, 0].reshape((originalValues.shape[0], 1))
        yNetworkValues = network(xOriginalValues)
        networkValues = np.concatenate((xOriginalValues, yNetworkValues), axis=1)
        self._original_line_series = self._chart_widget.create_line_series(
            originalValues, color=Qt.green
        )
        self._train_scatter_series = self._chart_widget.create_scatter_series(
            network_model.models[0].train.values, color=QColor(255, 0, 0)
        )
        self._network_line_series = self._chart_widget.create_line_series(
            networkValues, color=QColor(255, 165, 0)
        )
        self.setCentralWidget(self._chart_widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = NeuralNetworkWindow(NetworkModel())
    window.setWindowTitle(WINDOW_TITLE)
    window.show()
    window.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

    sys.exit(app.exec_())
