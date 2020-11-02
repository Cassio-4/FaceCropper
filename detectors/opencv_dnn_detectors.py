import cv2
import numpy as np


class OpenCV_DNN_Caffe_SSD:
    """
    https://towardsdatascience.com/face-detection-models-which-to-use-and-why-d263e82c302c
    It is a Caffe model which is based on the Single Shot-Multibox Detector (SSD)
    and uses ResNet-10 architecture as its backbone. It was introduced post OpenCV
    3.3 in its deep neural network module. There is also a quantized Tensorflow
    version that can be used but we will use the Caffe Model.
    """
    def __init__(self, model_file="weights/res10_300x300_ssd_iter_140000.caffemodel", config_file="weights/deploy.prototxt.txt"):
        self.net = cv2.dnn.readNetFromCaffe(config_file, model_file)

    def detect(self, img, confidence_threshold=0.5):

        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 117.0, 123.0))
        self.net.setInput(blob)
        faces = self.net.forward()

        scores = []
        boxes = []
        # Extracting boxes and scores
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                boxes.append(np.asarray((x, y, x1, y1), dtype=np.uint16))
                scores.append(confidence)
        return boxes, scores
