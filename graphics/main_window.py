import sys

from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow

__all__ = ['NeuralNetworkWindow']

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE = "Function approximation via neural network"


class NeuralNetworkWindow(QMainWindow):
    def __init__(self, parent=None):
        super(NeuralNetworkWindow, self).__init__(parent=parent)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = NeuralNetworkWindow()
    window.setWindowTitle(WINDOW_TITLE)
    window.show()
    window.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

    sys.exit(app.exec_())
