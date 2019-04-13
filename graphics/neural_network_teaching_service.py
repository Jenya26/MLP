from PyQt5.QtCore import Qt, QThread

__all__ = ['NeuralNetworkTeachingService']


class NeuralNetworkTeachingService(QThread):
    def __init__(self,
                 model,
                 teacher,
                 gradient,
                 error,
                 train_data_store,
                 batch=10,
                 learning_rate=1e-3,
                 iterations=1):
        QThread.__init__(self)
        self._teacher = teacher
        self._network = model
        self._gradient = gradient
        self._error = error
        self._train_data_store = train_data_store
        self._batch = batch
        self._learning_rate = learning_rate
        self._iterations = iterations
        self._on_update_model = None
        self._stop_callback = None
        self._is_stop = False

    def __del__(self):
        self.wait()

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, network):
        self._network = network

    @property
    def gradient(self):
        return self._gradient

    @gradient.setter
    def gradient(self, gradient):
        self._gradient = gradient

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, error):
        self._error = error

    @property
    def train_data_store(self):
        return self._train_data_store

    @train_data_store.setter
    def train_data_store(self, train_data_store):
        self._train_data_store = train_data_store

    @property
    def teacher(self):
        return self._teacher

    @teacher.setter
    def teacher(self, teacher):
        self._teacher = teacher

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._learning_rate = learning_rate

    @property
    def iterations(self):
        return self._iterations

    @iterations.setter
    def iterations(self, iterations):
        self._iterations = iterations

    @property
    def on_update_model(self):
        return self._on_update_model

    @on_update_model.setter
    def on_update_model(self, on_update_model):
        if not callable(on_update_model):
            raise ValueError('on_update_model should be callable')
        self._on_update_model = on_update_model

    @property
    def stop_callback(self):
        return self._stop_callback

    @stop_callback.setter
    def stop_callback(self, stop_callback):
        if not callable(stop_callback):
            raise ValueError('Stop callback should be callable')
        self._stop_callback = stop_callback

    def stop(self, stop_callback=None):
        self._is_stop = True
        if callable(stop_callback):
            self._stop_callback = stop_callback

    def run(self):
        self._is_stop = False
        while self._iterations > 0 and not self._is_stop:
            network = self._network.copy()
            self._teacher(
                network=network,
                gradient=self._gradient,
                error=self._error,
                data_store=self._train_data_store,
                max_iterations=1,
                batch=self._batch,
                learning_rate=self._learning_rate
            )
            self._network = network
            if self._on_update_model is not None:
                self._on_update_model(network)
            self._iterations -= 1
        if self._stop_callback is not None:
            self._stop_callback(self._iterations)
