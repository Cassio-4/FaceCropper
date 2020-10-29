import cv2
import numpy as np


def draw_boxes_on_image(image, trackable_objects):
    """
    Draws bounding boxes and Ids on image
    :param image: the image to draw on (using opencv)
    :param trackable_objects: an OrderedDict of TrackableObjects that exist
    :return: the drawn on frame
    """
    image_copy = image.copy()

    for num, to in zip(trackable_objects.keys(), trackable_objects.values()):
        box = to.bounding_box
        if to.disappeared_frames > 0:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.rectangle(image_copy, (box[0], box[1]), (box[2], box[3]), color, 1)
        cv2.putText(image_copy, str(num), (box[0], box[1]),
                    cv2.FONT_HERSHEY_PLAIN, 0.6, color, 2)
    return image_copy


def centroid_from_box(bbox):
    """
    :param bbox: numpy array of four numbers [x_left, y_top, x_right, y_bottom]
    :return: numpy array [x, y] of centroid coordinates
    """
    x = int((bbox[0] + bbox[2]) / 2.0)
    y = int((bbox[1] + bbox[3]) / 2.0)

    return np.asarray((x, y), dtype=np.uint16)


class TrackableObject:
    def __init__(self, object_id, bbox):
        self.id = object_id
        self.bounding_box = None
        self.centroid = None
        self.set_bounding_box(bbox)
        # Flag used to indicate if the object is still active on the image
        self.disappeared_frames = 0
        self.highest_detection = None
        self.highest_detection_score = 0.0
        self.centroid_when_registered = None

    def set_bounding_box(self, bbox):
        self.bounding_box = bbox
        self.centroid = centroid_from_box(bbox)

    def update_highest_detection(self, image, score):
        if score > self.highest_detection_score:
            self.highest_detection = image


class ProcessedEvent:
    """
    A wrapper for processed events that holds the event itself
    and its detections
    """
    def __init__(self, event, objects):
        self.event = event
        self.objects = objects
