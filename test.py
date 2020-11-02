from collections import OrderedDict
import config
import dlib
import cv2


def run_test():
    from utils.common import draw_boxes_on_image
    from imutils.video import FPS

    print("Opening video file")
    vc = cv2.VideoCapture(config.VIDEO_FILE)
    print("Opening video file: Done")
    frame = vc.read()
    frame = frame[1]

    # Initializing video writer, instantiated later if needed
    writer = None
    # Inicializando as dimensoes de video
    (H, W) = frame.shape[:2]

    # If results are to be saved on disc, instantiate writer
    if config.RECORD_TEST:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(config.RECORDING_PATH, fourcc, 30, (W, H), True)

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
            disappeared_objs.extend(config.CENTROID_TRACKER.update(rects, to, crops, scores))

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

        frame_drawn = draw_boxes_on_image(frame, to)
        # Show frame
        if config.SHOW_TEST:
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
