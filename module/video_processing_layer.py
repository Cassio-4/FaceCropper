from utils.common import ProcessedEvent, draw_boxes_on_image
from collections import OrderedDict
import config
import dlib
import cv2


def process_video(event):
    # Get file system path where event's video is stored
    path = event.fspath() + '/' + "%05d-capture.jpg"
    # Open video file
    config.LOGGER.Debug(1, "Opening video at {}".format(path))
    vc = cv2.VideoCapture(path)
    config.LOGGER.Debug(1, "Opening video at {}: Done".format(path))

    # Ordered dictionary of Trackable Objects
    to = OrderedDict()

    # Initialize number of processed frames
    total_frames = 0

    trackers = []
    disappeared_objs = []

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
    # the video processing phase, redirect them to the disappeared_objs
    # list, since there is no way to track them among events (yet?).
    for key in list(to.keys()):
        disappeared_objs.append(to.pop(key))

    # Return results
    return disappeared_objs, total_frames


def process_event(event):
    """
    Sends the event's video to be processed. Once returned, it checks event's
    info against the processing's results, evaluating if the results are
    reliable or not. If not, the parameter callback is True on return.

    :param event: The ZoneMinder event to be processed
    :return: a ProcessedEvent object and a callback boolean
    """
    config.LOGGER.Debug(1, "Processing {} 's video.".format(event.name()))
    disappeared_objs, total_frames = process_video(event)
    config.LOGGER.Debug(1, "Processing {} 's video: Done.".format(event.name()))

    # Grab the alarmed frame of this event
    alarmed_frame = cv2.imread(event.fspath() + '/' + "alarm.jpg", cv2.IMREAD_UNCHANGED)
    # Assume that the event does not need to be called back and check all cases
    # for something wrong
    callback = False
    if alarmed_frame is None and len(disappeared_objs) == 0:
        callback = True
        config.LOGGER.Error("Event {} has no objects and no alarm frame: Callback.".format(event.name()))
    if event.total_frames() < total_frames:
        callback = True
        config.LOGGER.Error("Event {} processed frames < total frames: Callback.".format(event.name()))

    config.LOGGER.log_detector(event.name(), len(disappeared_objs), event.total_frames(), total_frames)
    # Create and return a ProcessedEvent object, responsible for storing the
    # event along with the objects detected and their information
    return ProcessedEvent(event, disappeared_objs, alarmed_frame=alarmed_frame), callback


def process_event_batch(zm_events_batch):
    """
    Receives a list of zoneminder events and sends each one to be processed.
    Wrapping and returning the results.

    :param zm_events_batch: List of Event objects
    :return: batch_approved, batch_callback
    """
    # For each event in the list process it and save results
    batch_approved = []
    batch_callback = []
    for ev in zm_events_batch:
        config.LOGGER.Info("Processing event {}.".format(ev.name()))
        processed_ev, callback = process_event(ev)
        if callback:
            config.LOGGER.Info("Processing event {}: Callback.".format(ev.name()))
            batch_callback.append(processed_ev)
        else:
            config.LOGGER.Info("Processing event {}: Done.".format(ev.name()))
            batch_approved.append(processed_ev)
    # Return the report to main loop
    return batch_approved, batch_callback
