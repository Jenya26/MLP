import numpy as np
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QScatterSeries, QValueAxis
from PyQt5.QtGui import QPolygonF, QPainter
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from initializers import RangeInitializer

__all__ = ['NeuralNetworkChartWidget']

X_OFFSET = 1.  # percents
Y_OFFSET = 5.  # percents
POINTS_COUNT_ON_ONE_UNIT = 30


def series_to_polyline(xdata, ydata):
    """Convert series data to QPolygon(F) polyline

    This code is derived from PythonQwt's function named
    `qwt.plot_curve.series_to_polyline`"""
    size = len(xdata)
    dtype = np.float
    tinfo = np.finfo
    dtypesize = tinfo(dtype).dtype.itemsize

    polyline = QPolygonF(size)
    polyline_size = polyline.size()

    pointer = polyline.data()
    pointer_size = 2 * polyline_size * dtypesize
    pointer.setsize(pointer_size)

    memory = np.frombuffer(pointer, dtype)
    memory[:(size - 1) * 2 + 1:2] = xdata.T
    memory[1:(size - 1) * 2 + 2:2] = ydata.T

    return polyline


def get_x_data(data):
    return data[:, 0]


def get_y_data(data):
    return data[:, 1]


def init_series(series, color):
    pen = series.pen()
    pen.setColor(color)
    pen.setWidthF(5.)
    series.setPen(pen)
    series.setUseOpenGL(True)


def update_series(series, x_data, y_data):
    series.clear()
    polyline = series_to_polyline(x_data, y_data)
    series.append(polyline)


def update_axis(axis, data, offset):
    min = np.min(data)
    max = np.max(data)
    len = max - min
    offset = offset * (len / (100. - 2. * offset))
    axis.setRange(min - offset, max + offset)


def update_x_axes(chart, series, x_data):
    x_axis = chart.axisX(series)
    if x_axis is not None:
        update_axis(x_axis, x_data, X_OFFSET)


def update_y_axes(chart, series, y_data):
    y_axis = chart.axisY(series)
    if y_axis is not None:
        update_axis(y_axis, y_data, Y_OFFSET)


def update_axes(chart, series, x_data, y_data):
    update_x_axes(chart, series, x_data)
    update_y_axes(chart, series, y_data)


class NeuralNetworkChartWidget(QChartView):
    def __init__(self, title):
        self._chart = QChart()
        self._chart.legend().hide()
        self._chart.setTitle(title)
        super(NeuralNetworkChartWidget, self).__init__(self._chart)
        self.setRenderHint(QPainter.Antialiasing)
        self._range_initializer = RangeInitializer(-2., 2.)
        self._x_range = self._range_initializer((4 * POINTS_COUNT_ON_ONE_UNIT, 1))
        self._init_original_function()
        self._init_train_points()
        self._init_network_model()
        self._chart.createDefaultAxes()

    def _init_original_function(self):
        self._original_function = None
        self._original_line_series = QLineSeries()
        init_series(self._original_line_series, Qt.green)
        self._chart.addSeries(self._original_line_series)

    def _init_train_points(self):
        self._train_points = None
        self._train_scatter_series = QScatterSeries()
        init_series(self._train_scatter_series, QColor(255, 0, 0))
        self._chart.addSeries(self._train_scatter_series)

    def _init_network_model(self):
        self._network_model = None
        self._network_line_series = QLineSeries()
        init_series(self._network_line_series, QColor(255, 165, 0))
        self._chart.addSeries(self._network_line_series)

    @property
    def original_function(self):
        return self._original_function

    @original_function.setter
    def original_function(self, original_function):
        self._original_function = original_function
        y_data = self._original_function(self._x_range)
        update_series(
            self._original_line_series,
            self._x_range,
            y_data
        )
        update_axes(self._chart, self._original_line_series, self._x_range, y_data)

    @property
    def network_model(self):
        return self._network_model

    @network_model.setter
    def network_model(self, network_model):
        self._network_model = network_model
        update_series(
            self._network_line_series,
            self._x_range,
            self._network_model(self._x_range)
        )

    @property
    def train_points(self):
        return self._train_points

    @train_points.setter
    def train_points(self, train_points):
        x_data = get_x_data(train_points)
        y_data = get_y_data(train_points)
        self._train_points = train_points
        update_series(
            self._train_scatter_series,
            x_data,
            y_data
        )
