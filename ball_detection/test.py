import cv2
import numpy as np

cam = cv2.VideoCapture(1)

while True :
    ret, frame = cam.read()

    if ret :
        cv2.imshow("cam", frame)

    cv2.waitKey(1)

cv2.destroyAllWindows