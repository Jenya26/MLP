import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout


from models import NetworkModel
from graphics.neural_network_widget import NeuralNetworkWidget

__all__ = ['NeuralNetworkWindow']

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE = "Function approximation via neural network"


class NeuralNetworkWindow(QMainWindow):
    def __init__(self, network_model, parent=None):
        super(NeuralNetworkWindow, self).__init__(parent=parent)
        self._neural_network_widget = NeuralNetworkWidget(network_model, parent)
        self.setCentralWidget(self._neural_network_widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = NeuralNetworkWindow(NetworkModel())
    window.setWindowTitle(WINDOW_TITLE)
    window.show()
    window.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

    sys.exit(app.exec_())
