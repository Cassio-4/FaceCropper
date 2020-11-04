from pyzm.helpers.Monitor import Monitor
from datetime import datetime
from utils.grid import Grid
import json


class MyMonitor:
    def __init__(self, zm_monitor: Monitor):
        self.zm_monitor = zm_monitor
        self.grid = Grid((zm_monitor.dimensions()['height'], zm_monitor.dimensions()['width']), 10, 10)


class Logger:
    def __init__(self, debug=False, save_json=False):
        self.dtformat = "%b %d %Y %H:%M:%S.%f"
        self.debug = debug
        self.save_json = save_json

    def log_JSON(self, message, event_name=None, caller=None):
        if not self.save_json:
            return
        dt = datetime.now().strftime(self.dtformat)
        with open('output/json_log.txt', 'a+') as log:
            if event_name is None:
                log.write("[{}]\n".format(dt))
            else:
                log.write('[{}] from[{}]'.format(dt, event_name))
            log.write(json.dumps(message, indent=2))
            log.write("\n")

    def log_detector(self, event_name, obj_num, caller=None):
        dt = datetime.now().strftime(self.dtformat)
        with open('output/detection_log.txt', 'a+') as log:
            log.write('[{}]{} processed: {} objects found\n'. format(dt, event_name, obj_num))

    def Debug(self, level, message, caller=None):
        if self.debug:
            dt = datetime.now().strftime(self.dtformat)
            print('{} [DBG {}] {}'.format(dt, level, message))

    def Info(self, message, caller=None):
        dt = datetime.now().strftime(self.dtformat)
        print('{} [INF] {}'.format(dt, message))

    def Warning(self, message, caller=None):
        dt = datetime.now().strftime(self.dtformat)
        print('{}  [WAR] {}'.format(dt, message))

    def Error(self, message, caller=None):
        dt = datetime.now().strftime(self.dtformat)
        print('{} [ERR] {}'.format(dt, message))

    def Fatal(self, message, caller=None):
        dt = datetime.now().strftime(self.dtformat)
        print('{} [FAT] {}'.format(dt, message))
        exit(-1)

    def Panic(self, message, caller=None):
        dt = datetime.now().strftime(self.dtformat)
        print('{} [PNC] {}'.format(dt, message))
        exit(-2)
