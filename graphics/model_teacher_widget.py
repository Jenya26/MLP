from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton
from graphics.model_teacher_controller import ModelTeachingController

__all__ = ['ModelTeacherWidget']


class ModelTeacherWidget(QWidget):
    def __init__(self, network_model, parent=None):
        super(ModelTeacherWidget, self).__init__(parent)
        self._model_teaching_controller = ModelTeachingController(parent, network_model)
        self._model_teacher_layout = QHBoxLayout(self)

        self._start_button = QPushButton("Start", self)
        self._stop_button = QPushButton("Stop", self)

        self._model_teacher_layout.addWidget(self._stop_button)
        self._model_teacher_layout.addWidget(self._start_button)

        self._start_button.clicked.connect(self._start)
        self._stop_button.clicked.connect(self._model_teaching_controller.stop)

    def _start(self):
        controller = self._model_teaching_controller
        controller.iterations = 1000000000
        controller.start()
