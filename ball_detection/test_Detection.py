import cv2
import numpy as np
import pickle
import os
from ColorRange import *


def direction(frame, x, y, h, w) :
    xCenter = x + w // 2
    yCenter = y + h // 2
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(frame, (xCenter, yCenter), 2, (0, 255, 0), -1)
    print(xCenter, yCenter, sep = " ")


cam1 = cv2.VideoCapture('test.mp4')

range = load("color_range")
upper = range.upper
lower = range.lower

WhetherTHeFirstFrame = 0
while True :
    ret1, frame_now1 = cam1.read()

    #frame_compare1 = cv2.imread("ball_sample.jpg")
    #if WhetherTHeFirstFrame == 0 :
    #    frame_last1 = frame_now1
    #    WhetherTHeFirstFrame += 1
    #else :
    #    frame_compare1 = cv2.bitwise_xor(frame_now1, frame_last1)


    hsv1 = cv2.cvtColor(frame_now1, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv1, lower, upper)

    contours1, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours1 :
        area = cv2.contourArea(contour)
        if area > 30 :
            x, y, w, h = cv2.boundingRect(contour)
            direction(frame_now1, x, y, h, w)

    cv2.imshow("Camera 1", frame_now1)

    
    key = cv2.waitKey(50)
    if key == ord(" ") :
        break



cam1.release()
cv2.destroyAllWindows