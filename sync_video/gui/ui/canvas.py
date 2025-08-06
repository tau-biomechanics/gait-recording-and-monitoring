"""
Matplotlib canvas components for visualization in Qt
"""

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from ..utils import debug_log


class MatplotlibCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for plot embedding in Qt"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        debug_log("Creating MatplotlibCanvas")
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.setParent(parent)
        debug_log("MatplotlibCanvas created")