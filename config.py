from detectors.detectors_tensorflow import FaceboxesTensorflow
from utils.centroidtracker import CentroidTracker
import configparser

print("[Global] reading config file")
config = configparser.ConfigParser()
config.read('config.ini')
print("[Global] reading config file: Done")

# Instantiate face detector
print("[Global] Instantiating face detector")
print("[Global] Using FaceBoxes model.")
FACE_DETECTOR = FaceboxesTensorflow(config['FaceboxesTensorflow']['weights_path'])
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
    VIDEO_FILE = config['TestInfo']['video_file']
    RECORD_TEST = config['TestInfo'].getboolean('record_test')
    RECORDING_PATH = config['TestInfo']["recording_path"]
    SHOW_TEST = config['TestInfo'].getboolean('show')

else:
    # Mounting list of passive monitors from config file
    passive_monitors_string = config['ZMInfo']['passive_monitors_list'].split(',')
    PASSIVE_MONITORS = tuple([int(m) for m in passive_monitors_string])
    API_URL = config['ZMInfo']['apiurl']
    PORTAL_URL = config['ZMInfo']['portalurl']
    USER = config['ZMInfo']['user']
    POOLING_TIME = config['ZMInfo'].getfloat('pooling_time')
    SHOW_EVENT = config['ZMInfo'].getboolean('show_event_processing')
    SAVE_DETECTIONS = config['ZMInfo'].getboolean('save_detections')
    DELETE_PROCESSED_EVENTS = config['ZMInfo'].getboolean('delete_processed_events')

print("[Global] Setting global variables: Done")

# Clean trash
del config
del passive_monitors_string

