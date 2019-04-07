import numpy as np
from PyQt5.QtChart import QChart, QChartView, QLineSeries
from PyQt5.QtGui import QPolygonF, QPainter

__all__ = ['ChartWidget']


def series_to_polyline(data):
    """Convert series data to QPolygon(F) polyline

    This code is derived from PythonQwt's function named
    `qwt.plot_curve.series_to_polyline`"""
    size = len(data)
    polyline = QPolygonF(size)
    pointer = polyline.data()
    dtype, tinfo = np.float, np.finfo
    pointer.setsize(2 * polyline.size() * tinfo(dtype).dtype.itemsize)
    memory = np.frombuffer(pointer, dtype)
    xdata = data[:, 0]
    ydata = data[:, 1]
    memory[:(size - 1) * 2 + 1:2] = xdata
    memory[1:(size - 1) * 2 + 2:2] = ydata
    return polyline


class ChartWidget(QChartView):
    def __init__(self):
        self._chart = QChart()
        self._chart.legend().hide()
        super(ChartWidget, self).__init__(self._chart)
        self.setRenderHint(QPainter.Antialiasing)

    def draw_line_series(self, data, color=None):
        line_series = QLineSeries()
        pen = line_series.pen()
        if color is not None:
            pen.setColor(color)
        pen.setWidthF(.1)
        line_series.setPen(pen)
        line_series.setUseOpenGL(True)
        line_series.append(series_to_polyline(data))
        self._chart.addSeries(line_series)
        self._chart.createDefaultAxes()
