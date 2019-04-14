import sys

from PyQt5.QtWidgets import QApplication, QMainWindow

from graphics.default_neural_network_tabs import tabs
from graphics.neural_network_tabs_widget import NeuralNetworkTabsWidget

__all__ = ['NeuralNetworkWindow']

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE = "Function approximation via neural network"


class NeuralNetworkWindow(QMainWindow):
    def __init__(self, parent=None):
        super(NeuralNetworkWindow, self).__init__(parent=parent)
        self._neural_network_tabs_widget = NeuralNetworkTabsWidget(tabs)
        self.setCentralWidget(self._neural_network_tabs_widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = NeuralNetworkWindow()
    window.setWindowTitle(WINDOW_TITLE)
    window.show()
    window.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

    sys.exit(app.exec_())
