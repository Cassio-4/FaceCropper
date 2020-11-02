from pyzm.helpers.Monitor import Monitor
from utils.grid import Grid


class MyMonitor:
    def __init__(self, zm_monitor: Monitor):
        self.zm_monitor = zm_monitor
        self.grid = Grid((zm_monitor.dimensions()['height'], zm_monitor.dimensions()['width']), 10, 10)
