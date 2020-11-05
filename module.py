from requests.exceptions import Timeout, ConnectionError
from utils.pyzm_wrappers import MyMonitor
from utils.common import ProcessedEvent, draw_boxes_on_image
from collections import OrderedDict
import datetime
import requests
import base64
import config
import dlib
import json
import cv2


def process_passive_monitors(passive_monitors):
    config.LOGGER.Info("It's late, processing passive monitors")
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
    config.LOGGER.Info("Processing passive monitors: Done")


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
    config.LOGGER.Debug(1, "Opening event at {}".format(path))
    vc = cv2.VideoCapture(path)
    config.LOGGER.Debug(1, "Opening event at {}: Done".format(path))

    # Ordered dictionary of Trackable Objects
    to = OrderedDict()

    # Inicializar o numero de frames processados
    total_frames = 0

    trackers = []
    disappeared_objs = []
    alarmed_frame = None
    config.LOGGER.Info("Processing event {}.".format(event.name()))
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
        if total_frames % config.SKIP_FRAMES == 0:
            # Grab bounding boxes and scores returned by detector
            boxes, scores = config.FACE_DETECTOR.detect(rgb)
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
            disappeared_objs.extend(config.CENTROID_TRACKER.update(rects, to, crops, frame.copy(), scores))

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
            disappeared_objs.extend(config.CENTROID_TRACKER.update(rects, to))

        # save first alarmed frame?
        if total_frames == 3 or total_frames == 5:
            alarmed_frame = frame
        # Update total frames
        total_frames += 1

        if config.SHOW_EVENT:
            drawn_frame = draw_boxes_on_image(frame, to)
            cv2.imshow(event.name(), drawn_frame)
            _ = cv2.waitKey(1) & 0xFF
    config.CENTROID_TRACKER.reset_id_count()
    if config.SHOW_EVENT:
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
    config.LOGGER.Info("Processing event {}: Done.".format(event.name()))
    config.LOGGER.log_detector(event.name(), len(disappeared_objs))
    # Return the list with all the centroids info
    return ProcessedEvent(event, disappeared_objs, alarmed_frame=alarmed_frame)


def encode_image_base64(image):
    # Encode the highest scoring image in base64 to be sent in a JSON request
    # https://stackoverflow.com/questions/40928205/python-opencv-image-to-byte-string-for-json-transfer
    try:
        _, buffer = cv2.imencode('.png', image)
        b64_image = base64.b64encode(buffer)
        b64_image_with_prefix = "data:image/png;base64," + b64_image.decode("utf-8")
        return b64_image_with_prefix
    except cv2.error as e:
        config.LOGGER.Error(e)
        return None


def send_packet(packet):
    headers = {'content-type': 'application/json; charset=UTF-8'}
    config.LOGGER.log_JSON(packet)
    try:
        response = requests.post('https://www.atento.inf.br/api', data=json.dumps(packet),
                                 headers=headers, verify=False)
    except Timeout:
        config.LOGGER.Error("Request Timed Out.")
    except ConnectionError:
        config.LOGGER.Error("Connection error.")
    else:
        config.LOGGER.log_JSON(response.content.decode('utf-8'))


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
        # Fill the common parts of the request
        packet = {
            "requestNumber": 00,
            "companyCode": 4,
            "dispositiveType": 2,
            "captureDeviceCode": monitor_name,
            "appCode": 7,
            "latitude": "null",
            "longitude": "null",
            "truePictureTree": "99"
        }
        # If there were no detections, we will send an alarmed frame of this event
        if len(pe.objects) == 0:
            packet["flagFace"] = 0
            if pe.alarmed_frame is None:
                config.LOGGER.Error("Alarmed Frame not obtained for event {}\n".format(pe.event.name()))
            else:
                packet["trueImage"] = encode_image_base64(pe.alarmed_frame)
                if packet["trueImage"] is None:
                    config.LOGGER.Error("unable to encode image")
                    continue
                send_packet(packet)
        # If there were detections, flagFace = 1 and send image
        else:
            # For each detection in this event
            packet["flagFace"] = 1
            packet["personalType"] = 1
            for obj in pe.objects:
                # Add to JSON
                packet["trueImage"] = encode_image_base64(obj.highest_detection_frame)
                packet["cropFace"] = encode_image_base64(obj.highest_detection_crop)
                if packet["trueImage"] is None:
                    config.LOGGER.Error("unable to encode image")
                    continue
                # packet["atributesPerson"] = [{"atribute": "Age", "value": 31},
                #                            {"atribute": "Color", "value": 1},
                #                            {"atribute": "Sex", "value": "1"},
                #                            {"atribute": "Direction", "value": 2}]

                # Send it to ATENTO API
                send_packet(packet)

                if config.SAVE_DETECTIONS:
                    cv2.imwrite("output/{}{}.jpg".format(pe.event.name(), obj.id), obj.highest_detection_crop)
                    cv2.imwrite("output/trueImage{}{}.jpg".format(pe.event.name(), obj.id), obj.highest_detection_frame)

        if config.DELETE_PROCESSED_EVENTS:
            pe.event.delete()


def login_with_api():
    from getpass import getpass
    import pyzm.api as zmapi
    import traceback

    password = getpass("Please enter your password to access your Zoneminder server: ")
    api_options = {
        'apiurl': config.API_URL,
        'portalurl': config.PORTAL_URL,
        'user': config.USER,
        'password': password,
        'logger': config.LOGGER
    }
    if config.DISABLE_SSL:
        api_options['disable_ssl_cert_check'] = True

    config.LOGGER.Info('Running FaceCropper on {}'.format(api_options['apiurl']))

    # lets init the API
    try:
        zmapi = zmapi.ZMApi(options=api_options)
    except Exception as e:
        print('Error: {}'.format(str(e)))
        print(traceback.format_exc())
        exit(1)

    return zmapi


def update_event_filter(event_filter):
    now = datetime.datetime.now()
    ago = now - datetime.timedelta(minutes=12)
    event_filter['from'] = "{}-{}-{} {}:{}:{}".format(ago.year, ago.month, ago.day, ago.hour, ago.minute,
                                                      ago.second)
    event_filter['to'] = "{}-{}-{} {}:{}:{}".format(now.year, now.month, now.day, now.hour, now.minute,
                                                    now.second)


def run_module():
    from cachetools import TTLCache
    import schedule
    import pyzm
    import time

    config.LOGGER.Info('Using pyzm version: {}'.format(pyzm.__version__))
    zmapi = login_with_api()

    active_monitors = {}
    passive_monitors = {}
    # Get all monitors and build all monitor objects
    config.LOGGER.Info("Getting monitors")
    zm_monitors = zmapi.monitors()
    for m in zm_monitors.list():
        # print('Name:{} Enabled:{} Type:{} Dims:{}'.format(m.name(), m.enabled(), m.type(), m.dimensions()))
        # print(m.status())
        # Wrapping monitors along with their grids and separating according
        # to active and passive categories
        if m.id() in config.PASSIVE_MONITORS:
            config.LOGGER.Debug(1, 'Monitor: {} [id: {}] is a passive monitor.'.format(m.name(), m.id()))
            passive_monitors[m.id()] = MyMonitor(m)
        else:
            config.LOGGER.Debug(1, 'Monitor: {} [id: {}] is an active monitor.'.format(m.name(), m.id()))
            active_monitors[m.id()] = MyMonitor(m)
    config.LOGGER.Info("Getting monitors: Done")

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
    while True:
        config.LOGGER.Info("Pooling for events")
        time.sleep(config.POOLING_TIME)

        update_event_filter(event_filter)

        config.LOGGER.Info("Getting events for active monitors")
        event_batch = []
        # Get all events and create a batch
        for monitor in active_monitors.values():
            config.LOGGER.Debug(1, 'Getting events for {}.'.format(monitor.zm_monitor.name()))
            cam_events = monitor.zm_monitor.events(options=event_filter)
            for event in cam_events.list():
                if event.id() not in received_events_ids:
                    received_events_ids[event.id()] = event.id()
                    event_batch.append(event)
                    config.LOGGER.Debug(1, 'Event:{} Cause:{} Notes:{}'.format(event.name(),
                                                                               event.cause(), event.notes()))
        # Send batch to be processed
        config.LOGGER.Info("Batch has {} events, processing".format(len(event_batch)))
        batch_result = process_event_batch(event_batch)
        config.LOGGER.Info("Batch processing: Done")
        # Process the results of a batch
        process_batch_result(batch_result, active_monitors)
