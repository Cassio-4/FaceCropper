import cv2
import numpy as np


def draw_boxes_on_image(image, boxes, ids):
    image_copy = image.copy()

    for b, i in zip(boxes, ids):
        #ymin, xmin, ymax, xmax = b
        # b should be [left_x, top_y, right_x, bottom_y]
        cv2.rectangle(image_copy, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
        cv2.putText(image_copy, str(i), (b[0], b[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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
        self.active = True
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
