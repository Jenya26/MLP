from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLineEdit, QSlider, QComboBox
from graphics.neural_network_teaching_service import NeuralNetworkTeachingService

from gradients import Gradient
from gradients import ApproximateGradient

__all__ = ['NeuralNetworkTeachingControllerWidget']


class NeuralNetworkTeachingControllerWidget(QWidget):
    def __init__(self,
                 function_code,
                 model,
                 teacher,
                 gradient,
                 error,
                 train,
                 learning_rate,
                 parent=None):
        super(NeuralNetworkTeachingControllerWidget, self).__init__(parent)
        self._current_model_index = 0
        self._models = [model]

        self._model_teaching_service = NeuralNetworkTeachingService(
            model,
            teacher,
            gradient,
            error,
            train,
            learning_rate=learning_rate
        )
        self._model_teaching_service.start_callback = self.__start_callback
        self._model_teaching_service.stop_callback = self.__stop
        self._model_teaching_service._on_update_model = self.__on_update_model
        self._learning_rate = learning_rate

        self._on_change_model = None
        self._on_change_function = None
        self._on_stop_teaching = None
        self._on_start_teaching = None

        container = QVBoxLayout(self)
        container.addLayout(self._init_teacher_controller_ui(gradient))
        container.addLayout(self._init_teacher_history_ui(function_code))

        self._toggle_active_buttons(False)

    def _init_teacher_controller_ui(self, gradient):
        self._teacher_controller_ui = QHBoxLayout()

        self._start_button = QPushButton("Start")
        self._start_button.clicked.connect(self.start)

        self._stop_button = QPushButton("Stop")
        self._stop_button.clicked.connect(self._model_teaching_service.stop)

        self._iterations = 1000
        self._iterations_line_edit = QLineEdit(str(self._iterations))
        self._iterations_line_edit.setFixedWidth(120)
        self._iterations_line_edit.textChanged[str].connect(self._on_change_iterations)

        self._gradient_strategy_combo_box = QComboBox()
        self._gradient_strategy_combo_box.addItem("Gradient")
        self._gradient_strategy_combo_box.addItem("Approximate gradient")
        self._gradient_strategy_combo_box.setCurrentIndex(
            self.__get_current_gradient_strategy_index(gradient)
        )
        self._gradient_strategy_combo_box.currentIndexChanged.connect(self.__change_gradient_strategy_index)

        vbox = QVBoxLayout()
        vbox.addWidget(self._iterations_line_edit, alignment=Qt.Alignment())
        vbox.addWidget(self._gradient_strategy_combo_box, alignment=Qt.Alignment())

        self._teacher_controller_ui.addLayout(vbox)
        self._teacher_controller_ui.addWidget(self._start_button, alignment=Qt.Alignment())
        self._teacher_controller_ui.addWidget(self._stop_button, alignment=Qt.Alignment())

        return self._teacher_controller_ui

    def _init_teacher_history_ui(self, function_code):
        self._teacher_history_ui = QHBoxLayout()

        self._function_line_edit = QLineEdit(function_code)
        self._function_line_edit.setFixedWidth(120)
        self._function_line_edit.textChanged[str].connect(self.__on_change_function)

        self._learning_rate_line_edit = QLineEdit(str(self._learning_rate))
        self._learning_rate_line_edit.setFixedWidth(120)
        self._learning_rate_line_edit.textChanged[str].connect(self.__on_change_learning_rate)

        vbox = QVBoxLayout()
        vbox.addWidget(self._function_line_edit, alignment=Qt.AlignTop)
        vbox.addWidget(self._learning_rate_line_edit, alignment=Qt.AlignTop)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setValue(0)
        self._slider.setTickPosition(QSlider.NoTicks)
        self._slider.setTickInterval(1)
        self._slider.valueChanged[int].connect(self.__on_change_model)

        self._reset_button = QPushButton("Reset")
        self._reset_button.clicked.connect(self._reset_model_weights)

        self._teacher_history_ui.addLayout(vbox)
        self._teacher_history_ui.addWidget(self._slider, alignment=Qt.Alignment())
        self._teacher_history_ui.addWidget(self._reset_button, alignment=Qt.Alignment())

        return self._teacher_history_ui

    @property
    def on_change_model(self):
        return self._on_change_model

    @on_change_model.setter
    def on_change_model(self, on_change_model):
        if not callable(on_change_model):
            raise ValueError('on_update_model should be callable')
        self._on_change_model = on_change_model

    @property
    def on_change_function(self):
        return self._on_change_function

    @on_change_function.setter
    def on_change_function(self, on_change_function):
        if not callable(on_change_function):
            raise ValueError('on_change_function should be callable')
        self._on_change_function = on_change_function

    def __get_current_gradient_strategy_index(self, gradient):
        if type(gradient) == Gradient:
            return 0
        if type(gradient) == ApproximateGradient:
            return 1
        return -1

    def __change_gradient_strategy_index(self, selected):
        gradient = None
        if selected == 0:
            gradient = Gradient()
        if selected == 1:
            gradient = ApproximateGradient()
        if gradient is not None:
            self._model_teaching_service.gradient = gradient

    def __on_update_model(self, model):
        self._models += [model]
        if self._on_change_model is not None:
            self._on_change_model(model)
        self._current_model_index = len(self._models) - 1
        self._slider.setMaximum(self._current_model_index)
        self._slider.setValue(self._current_model_index)

    @property
    def current_model(self):
        index = self._current_model_index
        return self._models[index]

    def _reset_model_weights(self):
        model = self.current_model
        model.reset()

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

    def __on_change_learning_rate(self, text):
        is_parsed = False
        k = len(text)
        learning_rate = 0.
        while not is_parsed and k > 0:
            is_parsed = True
            try:
                learning_rate = float(text[:k])
            except ValueError:
                is_parsed = False
                k -= 1
        if not is_parsed:
            learning_rate = 0.
        l = 0
        value = learning_rate
        while is_parsed and text[l] != '.' and value == learning_rate:
            is_parsed = True
            l += 1
            try:
                value = float(text[l:k])
            except ValueError:
                is_parsed = False
        l -= 1
        self._learning_rate = learning_rate
        self._model_teaching_service.learning_rate = learning_rate
        self._learning_rate_line_edit.setText(str(text[l:k]))

    def __on_change_function(self, text):
        if self._on_change_function is not None:
            self._on_change_function(text)

    def _toggle_active_buttons(self, is_teaching):
        self._reset_button.setEnabled(not is_teaching)
        self._stop_button.setEnabled(is_teaching)
        self._start_button.setEnabled(not is_teaching)
        self._iterations_line_edit.setEnabled(not is_teaching)
        self._function_line_edit.setEnabled(not is_teaching)
        self._slider.setEnabled(not is_teaching)
        self._learning_rate_line_edit.setEnabled(not is_teaching)

    @property
    def on_start_teaching(self):
        return self._on_start_teaching

    @on_start_teaching.setter
    def on_start_teaching(self, on_start_teaching):
        if not callable(on_start_teaching):
            raise ValueError('on_start_teaching should be callable')
        self._on_start_teaching = on_start_teaching

    def __start_callback(self):
        if self._on_start_teaching is not None:
            self._on_start_teaching()

    def start(self, iterations=None):
        if type(iterations) is not int:
            iterations = self._iterations
        self._toggle_active_buttons(True)
        service = self._model_teaching_service
        service.iterations = iterations
        service.start()

    @property
    def on_stop_teaching(self):
        return self._on_stop_teaching

    @on_stop_teaching.setter
    def on_stop_teaching(self, on_stop_teaching):
        if not callable(on_stop_teaching):
            raise ValueError('on_stop_teaching should be callable')
        self._on_stop_teaching = on_stop_teaching

    def __stop(self, iterations, model):
        self._toggle_active_buttons(False)
        if self._on_stop_teaching is not None:
            self._on_stop_teaching(iterations, model)
