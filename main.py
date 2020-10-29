from Detectors.detectors_tensorflow import FaceboxesTensorflow
from zm_wrappers.pyzm_wrappers import MyMonitor
from centroidtracker import CentroidTracker
from collections import OrderedDict
from common import ProcessedEvent
from requests.exceptions import Timeout, ConnectionError
from getpass import getpass
import configparser
import requests
import base64
import json
import dlib
import cv2

print("[Global] reading config file")
config = configparser.ConfigParser()
config.read('config.ini')
print("[Global] reading config file: Done")
# Instantiate face detector
print("[Global] Instantiating face detector")
print("[Global] Using FaceBoxes model.")
face_detector = FaceboxesTensorflow(config['FaceboxesTensorflow']['weights_path'])
print("[Global] Instantiating face detector: Done")
# Instantiate Centroid Tracker for ID assignment
print("[Global] Instantiating centroid tracker: Done")
centroid_tracker = CentroidTracker(config['DEFAULT'].getint('max_disappeared'),
                                   config['DEFAULT'].getint('max_distance'))
print("[Global] Instantiating centroid tracker: Done")
SHOW_EVENT = config['DEFAULT'].getboolean('show_event_processing')
SAVE_DETECTIONS = config['DEFAULT'].getboolean('save_detections')


def process_passive_monitors(passive_monitors):
    print("It's late, processing passive monitors")
    event_filter = {
        'from': "24 hours ago",
        'object_only': False,
        'min_alarmed_frames': 1
    }

    event_batch = []
    # Get all events and create a batch
    for monitor in passive_monitors.values():
        print('Getting events for {}.'.format(monitor.zm_monitor.name()))
        cam_events = monitor.zm_monitor.events(options=event_filter)
        for event in cam_events.list():
            event_batch.append(event)
            print('Event:{} Cause:{} Notes:{}'.format(event.name(), event.cause(), event.notes()))
    # Send batch to be processed
    batch_result = process_event_batch(event_batch)
    process_batch_result(batch_result, passive_monitors)
    print("Processing passive monitors: Done")


def process_event_batch(zm_events_batch):
    # For each event in the list process it and save results
    batch_result = []
    for ev in zm_events_batch:
        processed_ev = process_event(ev)
        batch_result.append(processed_ev)
    # Return the report to main loop
    return batch_result


def process_event(event):
    # Get file system path where event is stored
    path = event.fspath() + '/' + event.video_file()
    # Open video file
    print("Opening event at {}".format(path))
    vc = cv2.VideoCapture(path)
    print("Opening event at {}: Done".format(path))

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

        # Update total frames
        total_frames += 1

        if SHOW_EVENT:
            cv2.imshow(event.name(), frame)
            _ = cv2.waitKey(1) & 0xFF
    centroid_tracker.reset_id_count()
    if SHOW_EVENT:
        cv2.destroyAllWindows()

    # Close video capture
    vc.release()

    # If there are still trackable objects that were not deregistered during
    # the event processing phase, redirect them to the disappeared_objs
    # list, since there is no way to track them among events.
    for key in list(to.keys()):
        disappeared_objs.append(to.pop(key))
    # With the event processed we have all the objects that were tracked,
    # their info, centroid data and highest scoring image.
    # Create a ProcessedEvent object, responsible for storing the event along
    # with the objects detected and their information
    print("Finished processing event {}.".format(event.name()))
    # Return the list with all the centroids info
    return ProcessedEvent(event, disappeared_objs)


def process_batch_result(batch_result, monitors):
    """
    Processes the results from a batch of events, being responsible for
    mounting and sending info to ATENTO

    :param batch_result: a list of ProcessedEvent objects
    :param monitors: a dict of monitors
    :return:
    """
    # For processed event in a batch result
    for pe in batch_result:
        # Grab this event's monitor name
        monitor_name = monitors.get(pe.event.monitor_id()).zm_monitor.name()
        # For each detection in this event
        for obj in pe.objects:
            # Encode the highest scoring image in base64 to be sent in a JSON request
            # https://stackoverflow.com/questions/40928205/python-opencv-image-to-byte-string-for-json-transfer
            _, buffer = cv2.imencode('.png', obj.highest_detection)
            b64_image = base64.b64encode(buffer)
            b64_image_with_prefix = "data:image/png;base64," + b64_image.decode("utf-8")
            # Build the JSON
            packet = {
                "requestNumber": 00,
                "companyCode": 4,
                "dispositiveType": 2,
                "captureDeviceCode": monitor_name,
                "appCode": 7,
                "latitude": "null",
                "longitude": "null",
                "personalType": 1,
                "flagFace": 1,
                "trueImage": b64_image_with_prefix,
                "truePictureTree": "99",
                # "atributesPerson": [{"atribute": "Age", "value": 31}, {"atribute": "Color", "value": 1},
                #                    {"atribute": "Sex", "value": "1"}, {"atribute": "Direction", "value": 2}]
            }
            # Send it to ATENTO API
            headers = {'content-type': 'application/json; charset=UTF-8'}
            try:
                response = requests.post('https://www.atento.inf.br/api', data=json.dumps(packet),
                                         headers=headers, verify=False)
            except Timeout:
                print("(process_batch_result()): request Timed Out.")
            except ConnectionError:
                print("(process_batch_result()): Connection error.")
            else:
                print(response.content)
            if SAVE_DETECTIONS:
                cv2.imwrite("output/{}{}.jpg".format(pe.event.name(), obj.id), obj.highest_detection)
        with open('output/log.txt', 'a+') as log:
            log.write("{} processed: {} objects found\n".format(pe.event.name(), len(pe.objects)))


def run_module():
    from cachetools import TTLCache
    import pyzm.api as zmapi
    import traceback
    import schedule
    import pyzm
    import time
    import datetime

    # Mounting list of passive monitors from config file
    passive_monitors_string = config['ZMInfo']['passive_monitors_list'].split(',')
    passive_monitors_ids = [int(m) for m in passive_monitors_string]

    print('Using pyzm version: {}'.format(pyzm.__version__))
    password = getpass("Please enter your password to access your Zoneminder server: ")
    api_options = {
        'apiurl': config['ZMInfo']['apiurl'],
        'portalurl': config['ZMInfo']['portalurl'],
        'user': config['ZMInfo']['user'],
        'password': password,
        'logger': None
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
    zm_monitors = zmapi.monitors()
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
        'from': None,
        'to': None,
        'object_only': False,
        'min_alarmed_frames': 1
    }

    # Before entering the main loop, schedule te passive monitors
    schedule.every().day.at("00:00").do(process_passive_monitors, passive_monitors)

    # This is a little cache to remember what events were
    # received in the previous calls, each cached id lasts for 15 minutes(900s)
    received_events_ids = TTLCache(maxsize=200, ttl=900)

    # =============== MAIN LOOP ===============
    print("Pooling for events")
    while True:
        time.sleep(config['ZMInfo'].getfloat('pooling_time'))
        now = datetime.datetime.now()
        ago = now - datetime.timedelta(minutes=12)
        event_filter['from'] = "{}-{}-{} {}:{}:{}".format(ago.year, ago.month, ago.day, ago.hour, ago.minute,
                                                          ago.second)
        event_filter['to'] = "{}-{}-{} {}:{}:{}".format(now.year, now.month, now.day, now.hour, now.minute,
                                                        now.second)
        print("Getting events for active monitors")
        event_batch = []
        # Get all events and create a batch
        for monitor in active_monitors.values():
            print('Getting events for {}.'.format(monitor.zm_monitor.name()))
            cam_events = monitor.zm_monitor.events(options=event_filter)
            for event in cam_events.list():
                if event.id() not in received_events_ids:
                    received_events_ids[event.id()] = event.id()
                    event_batch.append(event)
                    print('Event:{} Cause:{} Notes:{}'.format(event.name(), event.cause(), event.notes()))
        # Send batch to be processed
        print("Batch has {} events, processing".format(len(event_batch)))
        batch_result = process_event_batch(event_batch)
        print("Batch processing: Done")
        # Process the results of a batch
        process_batch_result(batch_result, active_monitors)


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

        frame_drawn = draw_boxes_on_image(frame, to)
        # Show frame
        if config['TestInfo'].getboolean('show'):
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
