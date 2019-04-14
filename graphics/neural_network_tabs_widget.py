from PyQt5.QtWidgets import QWidget, QTabWidget

from graphics.neural_network_widget import NeuralNetworkWidget

__all__ = ['NeuralNetworkTabsWidget']

CHART_TITLE = "Neural network"
CHART_UPDATE_INTERVAL = 200


class NeuralNetworkTabsWidget(QTabWidget):
    def __init__(self,
                 tabs=None,
                 parent=None):
        super(NeuralNetworkTabsWidget, self).__init__(parent=parent)
        # self._tab_widget = QTabWidget(self)
        for tab in tabs:
            neural_network_widget = NeuralNetworkWidget(
                tab['function'],
                tab['model'],
                tab['teacher'],
                tab['gradient'],
                tab['error'],
                tab['learning_rate']
            )
            self.addTab(neural_network_widget, "name")
