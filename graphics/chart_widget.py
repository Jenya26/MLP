import numpy as np
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QScatterSeries, QValueAxis
from PyQt5.QtGui import QPolygonF, QPainter

__all__ = ['ChartWidget']

X_OFFSET = 1.  # percents
Y_OFFSET = 5.  # percents


class ChartWidget(QChartView):
    def __init__(self, title):
        self._chart = QChart()
        self._chart.legend().hide()
        self._chart.setTitle(title)
        super(ChartWidget, self).__init__(self._chart)
        self.setRenderHint(QPainter.Antialiasing)

    @staticmethod
    def _series_to_polyline(xdata, ydata):
        """Convert series data to QPolygon(F) polyline

        This code is derived from PythonQwt's function named
        `qwt.plot_curve.series_to_polyline`"""
        size = len(xdata)
        dtype, tinfo = np.float, np.finfo

        polyline = QPolygonF(size)

        pointer = polyline.data()
        pointer.setsize(2 * polyline.size() * tinfo(dtype).dtype.itemsize)

        memory = np.frombuffer(pointer, dtype)
        memory[:(size - 1) * 2 + 1:2] = xdata
        memory[1:(size - 1) * 2 + 2:2] = ydata

        return polyline

    @staticmethod
    def _get_xy_data(data):
        return data[:, 0], data[:, 1]

    def update_series(self, series, data):
        x_data, y_data = self._get_xy_data(data)
        series.clear()
        polyline = self._series_to_polyline(x_data, y_data)
        series.append(polyline)

    def _update_x_axes(self, series, x_data):
        chart = self._chart
        x_axis = chart.axisX(series)
        x_min = np.min(x_data)
        x_max = np.max(x_data)
        x_norm = x_max - x_min
        x_offset = X_OFFSET * (x_norm / (100. - 2. * X_OFFSET))
        x_axis.setRange(x_min - x_offset, x_max + x_offset)

    def _update_y_axes(self, series, y_data):
        chart = self._chart
        y_axis = chart.axisY(series)
        y_min = np.min(y_data)
        y_max = np.max(y_data)
        y_norm = y_max - y_min
        y_offset = Y_OFFSET * (y_norm / (100. - 2. * Y_OFFSET))
        y_axis.setRange(y_min - y_offset, y_max + y_offset)

    def update_axes(self, series, data):
        x_data, y_data = self._get_xy_data(data)
        self._update_x_axes(series, x_data)
        self._update_y_axes(series, y_data)

    def create_line_series(self, data, color=None):
        x_data, y_data = self._get_xy_data(data)
        line_series = QLineSeries()
        pen = line_series.pen()
        if color is not None:
            pen.setColor(color)
        pen.setWidthF(5.)
        line_series.setPen(pen)
        line_series.setUseOpenGL(True)
        line_series.append(self._series_to_polyline(x_data, y_data))
        self._chart.addSeries(line_series)
        self._chart.createDefaultAxes()
        return line_series

    def create_scatter_series(self, data, color=None):
        x_data, y_data = self._get_xy_data(data)
        scatter_series = QScatterSeries()
        pen = scatter_series.pen()
        if color is not None:
            pen.setColor(color)
        pen.setWidthF(5.)
        scatter_series.setPen(pen)
        scatter_series.setUseOpenGL(True)
        scatter_series.append(self._series_to_polyline(x_data, y_data))
        self._chart.addSeries(scatter_series)
        self._chart.createDefaultAxes()
        return scatter_series
