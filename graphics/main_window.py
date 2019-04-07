import sys

import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt

from graphics.chart_widget import ChartWidget

__all__ = ['NeuralNetworkWindow']

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE = "Function approximation via neural network"


class NeuralNetworkWindow(QMainWindow):
    def __init__(self, parent=None):
        super(NeuralNetworkWindow, self).__init__(parent=parent)
        self._chart_widget = ChartWidget()
        self._chart_widget.draw_line_series(
            np.asarray([
                [-5., -5.],
                [-2., -2.],
                [-1., 1.],
                [2., 2.],
                [5., 5.]
            ]),
            color=Qt.red
        )
        self.setCentralWidget(self._chart_widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = NeuralNetworkWindow()
    window.setWindowTitle(WINDOW_TITLE)
    window.show()
    window.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

    sys.exit(app.exec_())
