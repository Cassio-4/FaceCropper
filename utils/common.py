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
        self.highest_detection_crop = None
        self.highest_detection_frame = None
        self.highest_detection_score = 0.0
        self.centroid_when_registered = None

    def set_bounding_box(self, bbox):
        self.bounding_box = bbox
        self.centroid = centroid_from_box(bbox)

    def update_highest_detection(self, face_crop, frame, score):
        if score > self.highest_detection_score:
            self.highest_detection_crop = face_crop
            self.highest_detection_frame = frame
            self.highest_detection_score = score


class ProcessedEvent:
    """
    A wrapper for processed events that holds the event itself
    and its detections
    """
    def __init__(self, event, objects, alarmed_frame=None):
        self.event = event
        self.objects = objects
        self.alarmed_frame = alarmed_frame
        self.__callback = False
        self.__keep_video = False

    def set_callback(self, state):
        """
        Sets the status of this ProcessedEvent. If True, the event will be considered
        incomplete and will not be inserted into cache. This way it will be received again on the next ZMAPI call.
        :param state: boolean representing the callback status of this event.
        :return:
        """
        self.__callback = bool(state)

    def get_callback_state(self):
        return self.__callback

    def set_keep_video(self, keep_video):
        """
        Checks and saves if this event is to be deleted from Zoneminder or not
        :param keep_video: Boolean
        :return:
        """
        if keep_video is True:
            self.__keep_video = keep_video

    def get_keep_video(self):
        return self.__keep_video

    def cleanup(self):
        """
        This method cleans the objects and alarmed_frame attributes, only call this once it
        has been appropriately processed and sent up to the API. I'm not sure if this actually saves
        memory during execution time or not, Hope it does but it clearly deserves more research
        :return:
        """
        del self.objects
        del self.alarmed_frame
