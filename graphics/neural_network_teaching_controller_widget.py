from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLineEdit, QComboBox, QSlider
from graphics.neural_network_teaching_service import NeuralNetworkTeachingService

__all__ = ['NeuralNetworkTeachingControllerWidget']


class NeuralNetworkTeachingControllerWidget(QWidget):
    def __init__(self, network_model, parent=None):
        super(NeuralNetworkTeachingControllerWidget, self).__init__(parent)
        self._network_model = network_model
        for model in self._network_model.models:
            model.subscribe_on_add_model(self._on_add_model)

        current_model = network_model.current_model
        self._model_teaching_controller = NeuralNetworkTeachingService(
             current_model.current_model,
             current_model.teacher,
             current_model.gradient,
             current_model.error,
             current_model.train
        )
        self._model_teaching_controller.stop_callback = self._stop
        self._model_teaching_controller._on_update_model = self._on_update_model

        container = QVBoxLayout(self)
        container.addLayout(self._init_teacher_controller_ui())
        container.addLayout(self._init_teacher_history_ui())

        self._toggle_active_buttons(False)

    def _init_teacher_controller_ui(self):
        self._teacher_controller_ui = QHBoxLayout()

        self._start_button = QPushButton("Start")
        self._start_button.clicked.connect(self._start)

        self._stop_button = QPushButton("Stop")
        self._stop_button.clicked.connect(self._model_teaching_controller.stop)

        self._iterations = 1000000
        self._iterations_line_edit = QLineEdit(str(self._iterations))
        self._iterations_line_edit.setFixedWidth(120)
        self._iterations_line_edit.textChanged[str].connect(self._on_change_iterations)

        self._teacher_controller_ui.addWidget(self._iterations_line_edit)
        self._teacher_controller_ui.addWidget(self._start_button)
        self._teacher_controller_ui.addWidget(self._stop_button)

        return self._teacher_controller_ui

    def _init_teacher_history_ui(self):
        self._teacher_history_ui = QHBoxLayout()

        self._models_list = QComboBox()
        items = [model.function_text for model in self._network_model.models]
        self._models_list.addItems(items)
        self._models_list.activated[str].connect(self._on_change_model)
        self._models_list.setCurrentIndex(self._network_model.current_model_index)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setValue(0)
        self._slider.setTickPosition(QSlider.NoTicks)
        self._slider.setTickInterval(1)
        self._slider.valueChanged[int].connect(self._on_change_network)

        self._reset_button = QPushButton("Reset")
        self._reset_button.clicked.connect(self._reset)

        self._teacher_history_ui.addWidget(self._models_list)
        self._teacher_history_ui.addWidget(self._slider)
        self._teacher_history_ui.addWidget(self._reset_button)

        return self._teacher_history_ui

    def _on_update_model(self, network):
        network_model = self._network_model
        current_model = network_model.current_model
        current_model.add_model(network)

    def _reset(self):
        self._network_model.current_model.current_model.reset()

    def _on_change_network(self, value):
        network_model = self._network_model
        current_model = network_model.current_model
        current_model.current_model_index = value

    def _on_add_model(self, model):
        network_model = self._network_model
        current_model = network_model.current_model
        self._slider.setMaximum(current_model.models_count - 1)
        self._slider.setValue(current_model.models_count - 1)

    def _on_change_model(self, text):
        network_model = self._network_model
        model_index = self._models_list.currentIndex()
        network_model.current_model_index = model_index
        current_model = network_model.current_model
        self._model_teaching_controller.network = current_model.current_model

    def _on_change_iterations(self, text):
        iterations = 0
        for ch in text:
            if ord('0') <= ord(ch) <= ord('9'):
                iterations = 10 * iterations + ord(ch) - ord('0')
        self._iterations = iterations
        self._iterations_line_edit.setText(str(iterations))

    def _toggle_active_buttons(self, is_teaching):
        self._reset_button.setEnabled(not is_teaching)
        self._models_list.setEnabled(not is_teaching)
        self._stop_button.setEnabled(is_teaching)
        self._start_button.setEnabled(not is_teaching)
        self._iterations_line_edit.setEnabled(not is_teaching)

    def _start(self):
        self._toggle_active_buttons(True)
        controller = self._model_teaching_controller
        controller.iterations = self._iterations
        controller.start()

    def _stop(self, iterations):
        self._toggle_active_buttons(False)
