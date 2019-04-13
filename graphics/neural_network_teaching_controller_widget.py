from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLineEdit, QSlider
from graphics.neural_network_teaching_service import NeuralNetworkTeachingService

__all__ = ['NeuralNetworkTeachingControllerWidget']


class NeuralNetworkTeachingControllerWidget(QWidget):
    def __init__(self,
                 model,
                 teacher,
                 gradient,
                 error,
                 train,
                 parent=None):
        super(NeuralNetworkTeachingControllerWidget, self).__init__(parent)
        self._current_model_index = 0
        self._models = [model]

        self._model_teaching_service = NeuralNetworkTeachingService(
             model,
             teacher,
             gradient,
             error,
             train
        )
        self._model_teaching_service.stop_callback = self._stop
        self._model_teaching_service._on_update_model = self.__on_update_model

        self._on_change_model = None

        container = QVBoxLayout(self)
        container.addLayout(self._init_teacher_controller_ui())
        container.addLayout(self._init_teacher_history_ui())

        self._toggle_active_buttons(False)
        self._model_teaching_service.start()

    def _init_teacher_controller_ui(self):
        self._teacher_controller_ui = QHBoxLayout()

        self._start_button = QPushButton("Start")
        self._start_button.clicked.connect(self._start)

        self._stop_button = QPushButton("Stop")
        self._stop_button.clicked.connect(self._model_teaching_service.stop)

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

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setValue(0)
        self._slider.setTickPosition(QSlider.NoTicks)
        self._slider.setTickInterval(1)
        self._slider.valueChanged[int].connect(self.__on_change_model)

        self._reset_button = QPushButton("Reset")
        self._reset_button.clicked.connect(self._reset_model_weights)

        self._teacher_history_ui.addWidget(self._slider)
        self._teacher_history_ui.addWidget(self._reset_button)

        return self._teacher_history_ui

    @property
    def on_change_model(self):
        return self._on_change_model

    @on_change_model.setter
    def on_change_model(self, on_change_model):
        if not callable(on_change_model):
            raise ValueError('on_update_model should be callable')
        self._on_change_model = on_change_model

    def __on_update_model(self, model):
        self._models += [model]
        if self._on_change_model is not None:
            self._on_change_model(model)
        self._current_model_index = len(self._models) - 1
        self._slider.setMaximum(self._current_model_index)
        self._slider.setValue(self._current_model_index)

    def _reset_model_weights(self):
        self._model.reset()

    def __on_change_model(self, value):
        models = self._models
        self._current_model_index = value
        if self._on_change_model is not None:
            self._on_change_model(models[value])

    def _on_change_iterations(self, text):
        iterations = 0
        for ch in text:
            if ord('0') <= ord(ch) <= ord('9'):
                iterations = 10 * iterations + ord(ch) - ord('0')
        self._iterations = iterations
        self._iterations_line_edit.setText(str(iterations))

    def _toggle_active_buttons(self, is_teaching):
        self._reset_button.setEnabled(not is_teaching)
        self._stop_button.setEnabled(is_teaching)
        self._start_button.setEnabled(not is_teaching)
        self._iterations_line_edit.setEnabled(not is_teaching)

    def _start(self):
        self._toggle_active_buttons(True)
        controller = self._model_teaching_service
        controller.iterations = self._iterations
        controller.start()

    def _stop(self, iterations):
        self._toggle_active_buttons(False)
