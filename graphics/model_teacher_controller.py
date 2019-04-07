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
        for i in range(10000000):
            window = self._window
            current_model = window.current_model
            learning_rate = current_model.learning_rate
            network = current_model.last_model
            gradient = current_model.gradient
            error = current_model.error
            train_data_store = current_model.train
            teacher = current_model.teacher
            teacher(
                network=network,
                gradient=gradient,
                error=error,
                data_store=train_data_store,
                max_iterations=10000,
                batch=10,
                learning_rate=learning_rate
            )
            window.update_network_chart()
