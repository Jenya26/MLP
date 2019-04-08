from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLineEdit
from graphics.model_teacher_controller import ModelTeachingController

__all__ = ['ModelTeacherWidget']


class ModelTeacherWidget(QWidget):
    def __init__(self, network_model, parent=None):
        super(ModelTeacherWidget, self).__init__(parent)
        self._model_teaching_controller = ModelTeachingController(parent, network_model)
        self._model_teaching_controller.stop_callback = self._stop
        self._model_teacher_layout = QHBoxLayout(self)

        self._start_button = QPushButton("Start", self)
        self._stop_button = QPushButton("Stop", self)

        self._iterations = 10000
        self._iterations_line_edit = QLineEdit(str(self._iterations), self)
        self._iterations_line_edit.setFixedWidth(120)
        self._iterations_line_edit.textChanged[str].connect(self._on_change_iterations)

        self._model_teacher_layout.addWidget(self._iterations_line_edit)
        self._model_teacher_layout.addWidget(self._start_button)
        self._model_teacher_layout.addWidget(self._stop_button)

        self._start_button.clicked.connect(self._start)
        self._stop_button.clicked.connect(self._model_teaching_controller.stop)

        self._toggle_active_buttons(False)

    def _on_change_iterations(self, text):
        iterations = 0
        for ch in text:
            if ord('0') <= ord(ch) <= ord('9'):
                iterations = 10 * iterations + ord(ch) - ord('0')
        self._iterations = iterations
        self._iterations_line_edit.setText(str(iterations))

    def _toggle_active_buttons(self, is_teaching):
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
