from PyQt5.QtCore import Qt, QThread

__all__ = ['ModelTeachingController']


class ModelTeachingController(QThread):
    def __init__(self, window, network_model):
        QThread.__init__(self)
        self._window = window
        self._network_model = network_model

    def __del__(self):
        self.wait()

    def run(self):
        for i in range(10):
            window = self._window
            current_model = window.current_model
            network = current_model.last_model
            gradient = current_model.gradient
            error = current_model.error
            train_data_store = current_model.train
            teacher = current_model.teacher
            teacher(network, gradient, error, train_data_store, 1000, 10)
            window.update_chart()

