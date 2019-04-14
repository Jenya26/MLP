from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLineEdit
from PyQt5.QtCore import Qt

__all__ = ['NeuralNetworkModelControllerWidget']


class NeuralNetworkModelControllerWidget(QWidget):
    def __init__(self,
                 model,
                 parent=None):
        super(NeuralNetworkModelControllerWidget, self).__init__(parent=parent)
        self._model = model
        self._layout = QVBoxLayout(self)

        for i, layer in enumerate(self._model.layers[:-1]):
            layer_count_line_edit = QLineEdit(str(layer.output_dimension))
            layer_count_line_edit.setFixedWidth(120)
            layer_count_line_edit.textChanged[str].connect(
                self.__on_change_layer_count_maker(layer_count_line_edit, i)
            )
            self._layout.addWidget(layer_count_line_edit, alignment=Qt.AlignTop)

    def __on_change_layer_count_maker(self, layer_count_line_edit, index):
        def __on_change_layer_count(text):
            layer_count = 0
            for ch in text:
                if ord('0') <= ord(ch) <= ord('9'):
                    layer_count = 10 * layer_count + ord(ch) - ord('0')
            layer_count_line_edit.setText(str(layer_count))
            layers = self._model.layers
            layer_count = max(layer_count, 1)
            layers[index].output_dimension = layer_count
            layers[index + 1].input_dimension = layer_count
        return __on_change_layer_count
