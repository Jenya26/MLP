from PyQt5.QtWidgets import QTabWidget

from graphics.neural_network_widget import NeuralNetworkWidget

__all__ = ['NeuralNetworkTabsWidget']


class NeuralNetworkTabsWidget(QTabWidget):
    def __init__(self,
                 tabs=None,
                 parent=None):
        super(NeuralNetworkTabsWidget, self).__init__(parent=parent)
        for tab in tabs:
            neural_network_widget = NeuralNetworkWidget(
                tab['function'],
                tab['model'],
                tab['teacher'],
                tab['gradient'],
                tab['error'],
                tab['learning_rate']
            )
            self.addTab(neural_network_widget, tab['function'])
            neural_network_widget.on_change_function = self.__on_change_function

    def __on_change_function(self, function_code):
        current_index = self.currentIndex()
        self.setTabText(current_index, function_code)
