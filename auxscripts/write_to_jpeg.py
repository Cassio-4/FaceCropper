# This is a simple script to write a video's frames in a .jpeg format
# required by zoneminder, so it can read from file as source
import cv2
import time


path = '/videos/p1e_s1_c1.avi'
vc = cv2.VideoCapture(path)
f = 0
while True:
    frame = vc.read()
    if frame is None or frame[0] is False:
        break
    #time.sleep(0.01)
    print(f)
    f += 1
    cv2.imwrite('p1e.jpeg', frame[1])
