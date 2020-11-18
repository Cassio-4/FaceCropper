from detectors.detectors_tensorflow import FaceboxesTensorflow
from utils.centroidtracker import CentroidTracker
from utils.pyzm_wrappers import Logger
import configparser

print("[Global] reading config file")
config = configparser.ConfigParser()
config.read('config.ini')
print("[Global] reading config file: Done")

# Instantiate face detector
print("[Global] Instantiating face detector")
print("[Global] Using FaceBoxes model.")
FACE_DETECTOR = FaceboxesTensorflow(config['FaceboxesTensorflow']['weights_path'],
                                    config['FaceboxesTensorflow'].getboolean('score_threshold'))
print("[Global] Instantiating face detector: Done")

# Instantiate Centroid Tracker for ID assignment
print("[Global] Instantiating centroid tracker: Done")
CENTROID_TRACKER = CentroidTracker(config['DEFAULT'].getint('max_disappeared'),
                                   config['DEFAULT'].getint('max_distance'))
print("[Global] Instantiating centroid tracker: Done")

# Create global constants
print("[Global] Setting global variables")
RUN_TEST = config['DEFAULT'].getboolean('run_test')
SKIP_FRAMES = config['DEFAULT'].getint('skip_frames')

if RUN_TEST:
    RECORD_TEST = config['TestInfo'].getboolean('record_test')
    SHOW_TEST = config['TestInfo'].getboolean('show')
    MONITOR_NAME = config['TestInfo']['monitor_name']
    MONITOR_ID = config['TestInfo'].getint('monitor_id')
    EVENT_PATH = config['TestInfo']['event_path']
    EVENT_NAME = config['TestInfo']['event_name']
    TOTAL_FRAMES = config['TestInfo'].getint('total_frames')
    EVENT_ID = config['TestInfo'].getint('event_id')

# Mounting list of passive monitors from config file
try:
    passive_monitors_string = config['ZMInfo']['passive_monitors_list'].split(',')
    PASSIVE_MONITORS = tuple([int(m) for m in passive_monitors_string])
except ValueError:
    PASSIVE_MONITORS = ()
API_URL = config['ZMInfo']['apiurl']
PORTAL_URL = config['ZMInfo']['portalurl']
USER = config['ZMInfo']['user']
DISABLE_SSL = config['ZMInfo'].getboolean('disable_ssl')
POOLING_TIME = config['ZMInfo'].getfloat('pooling_time')
SHOW_EVENT = config['ZMInfo'].getboolean('show_event_processing')
SAVE_DETECTIONS = config['Logger'].getboolean('save_detections')
DELETE_PROCESSED_EVENTS = config['ZMInfo'].getboolean('delete_processed_events')
LOGGER = Logger(config['Logger'].getboolean('debug'), config['Logger'].getboolean('save_json'))
print("[Global] Setting global variables: Done")

# Clean trash
del config
del passive_monitors_string

