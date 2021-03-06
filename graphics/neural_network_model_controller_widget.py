from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel, QComboBox
from PyQt5.QtCore import Qt

from functions import SigmoidFunction, LinearFunction

__all__ = ['NeuralNetworkModelControllerWidget']


def get_current_function_index(function):
    if isinstance(function, LinearFunction):
        return 0
    if isinstance(function, SigmoidFunction):
        return 1
    return -1


class NeuralNetworkModelControllerWidget(QWidget):
    def __init__(self,
                 model,
                 parent=None):
        super(NeuralNetworkModelControllerWidget, self).__init__(parent=parent)
        self._model = model
        self._layout = QVBoxLayout(self)
        self._layers_layout = QVBoxLayout()

        self.__update_layers_info()
        add_layout_button = QPushButton("Add new layer")
        add_layout_button.clicked.connect(self._add_layer)
        self._layout.addLayout(self._layers_layout, 1)
        self._layout.addWidget(add_layout_button, alignment=Qt.AlignBottom)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.__update_layers_info()

    def __update_layers_info(self):
        for i in range(self._layers_layout.count()):
            child = self._layers_layout.itemAt(i)
            widget = child.widget()
            widget.setVisible(False)
        for i, layer in enumerate(self._model.layers[:-1]):
            self.__add_layer_info(i, layer)
        index = len(self._model.layers) - 1
        self.__add_layer_info(index, self._model.layers[-1], False)
        self._layers_layout.setAlignment(Qt.AlignTop)

    def __create_new_layer(self, i, layer, enabled=True):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layer_count_line_edit = QLineEdit(str(layer.output_dimension))
        layer_count_line_edit.setFixedWidth(120)
        layer_count_line_edit.textChanged[str].connect(
            self.__on_change_layer_count_maker(layer_count_line_edit, i)
        )
        layer_count_line_edit.setEnabled(enabled)
        remove_layout_button = QPushButton("X")
        remove_layout_button.clicked.connect(self._remove_layout_maker(i))
        remove_layout_button.setVisible(enabled)
        label = QLabel("Layer %d:" % (i + 1))
        line = QHBoxLayout()
        line.addWidget(label, alignment=Qt.AlignTop)
        line.addWidget(remove_layout_button, alignment=Qt.AlignTop)
        activation_functions = QComboBox()
        activation_functions.addItem("Linear function")
        activation_functions.addItem("Sigmoid function")
        activation_functions.setCurrentIndex(get_current_function_index(layer.activation_function))
        activation_functions.currentIndexChanged.connect(self.__change_activation_function_maker(i))
        layout.addLayout(line)
        layout.addWidget(layer_count_line_edit, alignment=Qt.AlignTop)
        layout.addWidget(activation_functions, alignment=Qt.AlignTop)
        return widget

    def __add_layer_info(self, i, layer, enabled=True):
        item = self._layers_layout.itemAt(i)
        if item is not None:
            widget = item.widget()
            widget_layout = widget.layout()
            line_item = widget_layout.itemAt(0)
            line = line_item.layout()
            remove_layout_button_item = line.itemAt(1)
            remove_layout_button = remove_layout_button_item.widget()
            remove_layout_button.setVisible(enabled)
            layer_count_line_edit_item = widget_layout.itemAt(1)
            layer_count_line_edit = layer_count_line_edit_item.widget()
            layer_count_line_edit.setText(str(layer.output_dimension))
            layer_count_line_edit.setEnabled(enabled)
            activation_function_item = widget_layout.itemAt(2)
            activation_function = activation_function_item.widget()
            activation_function.setCurrentIndex(get_current_function_index(layer.activation_function))
            widget.setVisible(True)
        else:
            widget = self.__create_new_layer(i, layer, enabled)
            self._layers_layout.addWidget(widget, alignment=Qt.AlignTop)

    def __change_activation_function_maker(self, index):
        def __change_activation_function(selected):
            function = None
            if selected == 0:
                function = LinearFunction()
            if selected == 1:
                function = SigmoidFunction()
            if function is not None:
                self._model.change_activation_function(index, function)
        return __change_activation_function

    def _remove_layout_maker(self, index):
        def _remove_layout():
            self._model.remove_layer(index)
            self.__update_layers_info()
            self.repaint()
        return _remove_layout

    def _add_layer(self):
        self._model.add_layer()
        self.__update_layers_info()

    def __on_change_layer_count_maker(self, layer_count_line_edit, index):
        def __on_change_layer_count(text):
            layer_count = 0
            for ch in text:
                if ord('0') <= ord(ch) <= ord('9'):
                    layer_count = 10 * layer_count + ord(ch) - ord('0')
            layer_count_line_edit.setText(str(layer_count))
            layer_count = max(layer_count, 1)
            self._model.change_layer_count(index, layer_count)
        return __on_change_layer_count
