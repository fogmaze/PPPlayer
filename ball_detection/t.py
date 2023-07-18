import cv2
import numpy as np

cap = cv2.VideoCapture(0)\

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vr = cv2.VideoWriter("test.mp4", fourcc, 30, (640, 480))

while True:
    ret, frame = cap.read()
    vr.write(frame)
    cv2.imshow("a", frame)
    k = cv2.waitKey(1)
    if k == ord(' ') :
        break

vr.release()
