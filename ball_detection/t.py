import cv2
import numpy as np

cap = cv2.VideoCapture(0)\

while True :
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(5)
