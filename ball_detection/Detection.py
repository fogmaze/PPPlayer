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


cam1 = cv2.VideoCapture(0)
cam2 = cv2.VideoCapture(1)

range = load("color_range")
upper = range.upper
lower = range.lower

WhetherTHeFirstFrame = 0
while True :
    ret1, frame_now1 = cam1.read()
    ret2, frame_now2 = cam2.read()

    frame_compare1 = cv2.imread("ball_sample.jpg")
    frame_compare2 = cv2.imread("ball_sample.jpg")
    if WhetherTHeFirstFrame == 0 :
        frame_last1 = frame_now1
        frame_last2 = frame_now2
        WhetherTHeFirstFrame += 1
    else :
        frame_compare1 = cv2.bitwise_xor(frame_now1, frame_last1)
        frame_compare2 = cv2.bitwise_xor(frame_now2, frame_last2)


    hsv1 = cv2.cvtColor(frame_compare1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame_compare2, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv1, lower, upper)
    mask2 = cv2.inRange(hsv2, lower, upper)

    contours1, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours1 :
        area = cv2.contourArea(contour)
        if area > 200 :
            x, y, w, h = cv2.boundingRect(contour)
            direction(frame_now1, x, y, h, w)

    for contour in contours2 :
        area = cv2.contourArea(contour)
        if area > 200 :
            x, y, w, h = cv2.boundingRect(contour)
            direction(frame_now2, x, y, h, w)

    cv2.imshow("Camera 1", frame_now1)
    cv2.imshow("Camera 2", frame_now2)

    
    key = cv2.waitKey(5)
    if key == ord(" ") :
        break



cam1.release()
cam2.release()
cv2.destroyAllWindows