import time
from PyQt5.QtCore import Qt, QThread

__all__ = ['ModelTeachingController']


class ModelTeachingController(QThread):
    def __init__(self, window, network_model, iterations=1, ):
        QThread.__init__(self)
        self._window = window
        self._network_model = network_model
        self._last_update_time = time.time()
        self._iterations = iterations
        self._stop_callback = None
        self._is_stop = False

    def __del__(self):
        self.wait()

    @property
    def stop_callback(self):
        return self._stop_callback

    @stop_callback.setter
    def stop_callback(self, stop_callback):
        if not callable(stop_callback):
            raise ValueError('Stop callback should be callable')
        self._stop_callback = stop_callback

    @property
    def iterations(self):
        return self._iterations

    @iterations.setter
    def iterations(self, iterations):
        self._iterations = iterations

    def stop(self, stop_callback=None):
        self._is_stop = True
        if callable(stop_callback):
            self._stop_callback = stop_callback

    def run(self):
        window = self._window
        current_model = window.current_model
        learning_rate = current_model.learning_rate
        network = current_model.last_model
        gradient = current_model.gradient
        error = current_model.error
        train_data_store = current_model.train
        teacher = current_model.teacher
        self._is_stop = False
        while self._iterations > 0 and not self._is_stop:
            teacher(
                network=network,
                gradient=gradient,
                error=error,
                data_store=train_data_store,
                max_iterations=1,
                batch=10,
                learning_rate=learning_rate
            )
            current_time = time.time()
            if current_time - self._last_update_time > .5:
                window.update_network_chart()
                self._last_update_time = time.time()
            self._iterations -= 1
        window.update_network_chart()
        if self._stop_callback is not None:
            print(self._stop_callback)
            self._stop_callback(self._iterations)
