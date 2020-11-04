import base64
import cv2

img_data = ''

with open("output/imageToSave.png", "wb") as fh:
    fh.write(base64.decodebytes(img_data))
