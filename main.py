from zm_wrappers.pyzm_wrappers import MyMonitor
from centroidtracker import CentroidTracker
from collections import OrderedDict
from grid import Grid
import configparser
import dlib
import cv2

print("[Global] reading config file")
config = configparser.ConfigParser()
config.read('config.ini')
print("[Global] reading config file: Done")
# Instantiate face detector
print("[Global] Instantiating face detector")
if config['DEFAULT']['detector'] == 'FaceBoxes':
    from Detectors.detectors_tensorflow import FaceboxesTensorflow
    print("[Global] Using FaceBoxes model.")
    face_detector = FaceboxesTensorflow(config['FaceboxesTensorflow']['weights_path'])
else:
    from Detectors.opencv_dnn_detectors import OpenCV_DNN_Caffe_SSD
    print("[Global] Using Caffe OpencvDNN.")
    face_detector = OpenCV_DNN_Caffe_SSD(config['OpenCVDNNCaffeModel']['weights_path'],
                                         config['OpenCVDNNCaffeModel']['model_conf'])
print("[Global] Instantiating face detector: Done")
# Instantiate Centroid Tracker for ID assignment
print("[Global] Instantiating centroid tracker: Done")
centroid_tracker = CentroidTracker(max_disappeared=40, max_distance=50)
print("[Global] Instantiating centroid tracker: Done")


def process_event_batch(zm_events_batch):
    # Start two dicts to keep this batch's results
    batch_results_reg = {}
    batch_results_dereg = {}
    # For each event in the list process it and save results
    for ev in zm_events_batch:
        found, lost = process_event(ev)
        # If this id has never been used before then create an empty list
        # in the dictionary where to store centroids
        if batch_results_reg.get(ev.monitor_id()) is None or \
                batch_results_dereg.get(ev.monitor_id()) is None:
            batch_results_dereg[ev.monitor_id()] = []
            batch_results_reg[ev.monitor_id()] = []
        batch_results_reg[ev.monitor_id()].extend(found)
        batch_results_dereg[ev.monitor_id()].extend(lost)
    # Return the report to main loop so we can write results on to the grid
    return batch_results_reg, batch_results_dereg


def process_event(event):
    # Get file system path where event is stored
    path = event.fspath()
    # Open video file
    print("Opening event at {}".format(path))
    vc = cv2.VideoCapture(path)
    print("Opening event at {}: Done".format(path))
    frame = vc.read()
    frame = frame[1]

    # Getting video dimensions
    # (H, W) = frame.shape[:2]

    # Ordered dictionary of Trackable Objects
    to = OrderedDict()

    # Inicializar o numero de frames processados
    total_frames = 0

    trackers = []
    disappeared_objs = []
    print("Processing event {}.".format(event.name()))
    # Comecar a ler os frames
    while True:
        frame = vc.read()
        frame = frame[1]
        # Se estamos processando um video e nao recebemos um frame, chegamos ao fim
        if frame is None:
            break
        # Convert frame to RGB format required by dlib and Faceboxes
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rects = []

        # If 'skip_frames' passed, run detector
        if total_frames % config['DEFAULT'].getint('skip_frames') == 0:
            # Grab bounding boxes and scores returned by detector
            boxes, scores = face_detector.detect(rgb)
            trackers = []
            crops = []

            for box, score in zip(boxes, scores):
                (left_x, top_y, right_x, bottom_y) = box.astype("int")
                # Construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                rect = dlib.rectangle(left_x, top_y, right_x, bottom_y)
                # Create a correlation tracker and start tracking
                tracker = dlib.correlation_tracker()
                tracker.start_track(rgb, rect)
                trackers.append(tracker)
                # Add the bounding box coordinates to the rectangles list
                rects.append((left_x, top_y, right_x, bottom_y))
                crop_img = frame[top_y-5:bottom_y+5, left_x-5:right_x+5]
                crops.append(crop_img)

            # Update each object based on recent detections
            disappeared_objs.extend(centroid_tracker.update(rects, to, crops, scores))

        # Else, update trackers
        else:
            for tracker in trackers:
                # Update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()
                # Unpack the position object
                left_x = int(pos.left())
                top_y = int(pos.top())
                right_x = int(pos.right())
                bottom_y = int(pos.bottom())
                # Add the bounding box coordinates to the rectangles list
                rects.append((left_x, top_y, right_x, bottom_y))

            # Update each object based on recent detections
            disappeared_objs.extend(centroid_tracker.update(rects, to))

        # Update total frames
        total_frames += 1

    # If there are still trackable objects that were not deregistered during
    # the event processing phase, redirect them to the disappeared_objs
    # list, since there is no way to track them among events.
    for key in to.keys():
        disappeared_objs.append(to.pop(key))
    # With the event processed we have all the objects that were tracked,
    # their info, centroid data and highest scoring image.
    # Now, to finish processing them and return the centroids to
    # annotate on the grid

    # First save/send the highest scoring image to ATENTO
    # Todo study protocol to send image and finding the highest score image
    #cv2.imwrite()

    # Now, build two lists with centroid info of where the object was first
    # registered and where it was deregistered
    lost = []
    found = []
    for i, o in enumerate(disappeared_objs):
        found.append(o.centroid_when_registered)
        lost.append(o.centroid)
       # cv2.imwrite('videos/output/{}{}.jpg'.format(event.name(), i))

    # Close video capture
    vc.release()
    print("Finished processing event {}.".format(event.name()))
    # Return the list with all the centroids info
    return found, lost


def run_module():
    import pyzm
    import pyzm.api as zmapi
    import traceback
    import time

    use_zmlog = False
    has_zmlog = False

    # Mounting list of passive monitors from config file
    passive_monitors_string = config['ZMInfo']['passive_monitors_list'].split(',')
    passive_monitors_ids = [int(m) for m in passive_monitors_string]

    print('Using pyzm version: {}'.format(pyzm.__version__))
    if use_zmlog:
        try:
            import pyzm.ZMLog as zmlog  # only if you want to log to ZM
            has_zmlog = True
        except ImportError as e:
            print('Could not import ZMLog, function will be disabled:' + str(e))
            zmlog = None

    # This is for logging in to ZM
    print("Initing Log")
    zm_log_override = {
        'log_level_syslog': 3,
        'log_level_db': -5,
        'log_debug': 1,
        # 'log_level_file': -5,
        'log_debug_target': None
    }
    # Todo Having issues with logging initialization
    if has_zmlog:
        zmlog.init(name='FaceCropper', override=zm_log_override)
        print("Initing Log: Done")
    # Todo Ask user to input password, instead of having it on a .ini file (security reasons, obviously)
    api_options = {
        'apiurl': config['ZMInfo']['apiurl'],
        'portalurl': config['ZMInfo']['portalurl'],
        'user': config['ZMInfo']['user'],
        'password': config['ZMInfo']['password'],
        'logger': None,  # use none if you don't want to log to ZM,
        # 'disable_ssl_cert_check': True
    }

    print('Running FaceCropper on {}'.format(api_options['apiurl']))
    # lets init the API
    try:
        zmapi = zmapi.ZMApi(options=api_options)
    except Exception as e:
        print('Error: {}'.format(str(e)))
        print(traceback.format_exc())
        exit(1)

    active_monitors = {}
    passive_monitors = {}
    # Get all monitors and build all monitor objects
    print("Getting monitors")
    zm_monitors = zmapi.Monitors()
    for m in zm_monitors.list():
        print('Name:{} Enabled:{} Type:{} Dims:{}'.format(m.name(), m.enabled(), m.type(), m.dimensions()))
        print(m.status())
        # Wrapping monitors along with their grids and separating according
        # to active and passive categories
        if m.id() in passive_monitors_ids:
            print('Monitor: {} [id: {}] is a passive monitor.'.format(m.name(), m.id()))
            passive_monitors[m.id()] = MyMonitor(m)
        else:
            print('Monitor: {} [id: {}] is an active monitor.'.format(m.name(), m.id()))
            active_monitors[m.id()] = MyMonitor(m)
    print("Getting monitors: Done")

    # Starting the events filter
    event_filter = {
        'from': '24 hours ago',
        'object_only': False,
        'min_alarmed_frames': 1,
        'max_events': 5,
    }

    # Starting auxiliary structures
    received_events_ids = []

    print("Pooling for events")
    start_time = time.time()
    while True:
        time.sleep(60.0 - ((time.time() - start_time) % 60.0))
        print("Getting events for active monitors")
        # TODO alter filter accordingly
        # event_filter['from'] = something
        event_batch = []
        # Get all events and create a batch
        for monitor in active_monitors.values():
            print('Getting events for {}.'.format(monitor.zm_monitor.name()))
            cam_events = monitor.zm_monitor.events(options=event_filter)
            for event in cam_events.list():
                event_batch.append(event)
                print('Event:{} Cause:{} Notes:{}'.format(event.name(), event.cause(), event.notes()))
        # Send batch to be processed
        process_event_batch(event_batch)
        # TODO Grab batch results and send to respective Grid
        zmlog.close()


def run_test():
    from common import draw_boxes_on_image
    from imutils.video import FPS

    print("Opening video file")
    vc = cv2.VideoCapture(config['TestInfo']['video_file'])
    print("Opening video file: Done")
    frame = vc.read()
    frame = frame[1]

    # Initializing video writer, instantiated later if needed
    writer = None
    # Inicializando as dimensoes de video
    (H, W) = frame.shape[:2]

    # If results are to be saved on disc, instantiate writer
    if config['TestInfo'].getboolean('record_test'):
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(config['TestInfo']["recording_path"], fourcc, 30, (W, H), True)

    # Instanciar o rastreador de centroides
    ct = CentroidTracker(max_disappeared=40, max_distance=50)
    # Instantiate grid
    grid = Grid((H, W))
    # Ordered dictionary of Trackable Objects
    to = OrderedDict()

    # Inicializar o numero de frames processados
    total_frames = 0

    trackers = []
    disappeared_objs = []

    # Start the fps counter
    fps = FPS().start()
    # Comecar a ler os frames
    while True:
        frame = vc.read()
        frame = frame[1]
        # Se estamos processando um video e nao recebemos um frame, chegamos ao fim
        if frame is None:
            break
        # Convert frame to RGB format required by dlib and Faceboxes
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rects = []

        # If 'skip_frames' passed, run detector
        if total_frames % config['DEFAULT'].getint('skip_frames') == 0:
            # Grab bounding boxes and scores returned by detector
            boxes, scores = face_detector.detect(rgb)
            trackers = []
            crops = []

            for box, score in zip(boxes, scores):
                (left_x, top_y, right_x, bottom_y) = box.astype("int")
                # Construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                rect = dlib.rectangle(left_x, top_y, right_x, bottom_y)
                # Create a correlation tracker and start tracking
                tracker = dlib.correlation_tracker()
                tracker.start_track(rgb, rect)
                trackers.append(tracker)
                # Add the bounding box coordinates to the rectangles list
                rects.append((left_x, top_y, right_x, bottom_y))
                crop_img = frame[top_y - 5:bottom_y + 5, left_x - 5:right_x + 5]
                crops.append(crop_img)

            # Update each object based on recent detections
            disappeared_objs.extend(centroid_tracker.update(rects, to, crops, scores))

        # Else, update trackers
        else:
            for tracker in trackers:
                # Update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()
                # Unpack the position object
                left_x = int(pos.left())
                top_y = int(pos.top())
                right_x = int(pos.right())
                bottom_y = int(pos.bottom())
                # Add the bounding box coordinates to the rectangles list
                rects.append((left_x, top_y, right_x, bottom_y))

            # Update each object based on recent detections
            disappeared_objs.extend(centroid_tracker.update(rects, to))

        # Show frame
        if config['TestInfo'].getboolean('show'):
            frame_drawn = draw_boxes_on_image(frame, [b.bounding_box for b in to.values()], to.keys())
            cv2.imshow("Frame", frame_drawn)

        # Check if we're writing the frame on disc
        if writer is not None:
            writer.write(frame_drawn)

        key = cv2.waitKey(1) & 0xFF
        # If 'q' is pressed, quit
        if key == ord("q"):
            break
        total_frames += 1
        fps.update()

    # If there are still trackable objects that were not deregistered during
    # the event processing phase, redirect them to the disappeared_objs
    # list, since there is no way to track them among events.
    for key in to.keys():
        disappeared_objs.append(to.pop(key))
    # With the event processed we have all the objects that were tracked,
    # their info, centroid data and highest scoring image.
    # Now, to finish processing them and return the centroids to
    # annotate on the grid
    # First save/send the highest scoring image to ATENTO
    for i, det in enumerate(disappeared_objs):
        if not (det.highest_detection is None):
            cv2.imwrite("videos/output/{}.jpg".format(str(i)), det.highest_detection)

    # Parar o contador e mostar informacoes de FPS
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    print("[INFO] total frames: {:.2f}".format(total_frames))

    # Close writer if needed
    if writer is not None:
        writer.release()
    # Close video capture
    vc.release()
    # Close all windows
    cv2.destroyAllWindows()


if __name__ == "__main__":

    if config['DEFAULT'].getboolean('run_test'):
        print("[MAIN] Running test")
        run_test()
    else:
        print("[MAIN] Starting FaceCropper module")
        run_module()
