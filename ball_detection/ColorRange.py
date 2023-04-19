import cv2
import numpy as np
import pickle

class ColorRange :
    
    def __init__(self, ) :

def empty(a) :
    pass

cam1 = cv2.VideoCapture(1)

cv2.namedWindow("ColorRange")
cv2.resizeWindow("ColorRange", 640, 640)

cv2.createTrackbar("Hue Min", "ColorRange", 0, 179, empty)
cv2.createTrackbar("Hue Max", "ColorRange", 179, 179, empty)
cv2.createTrackbar("Sat Min", "ColorRange", 0, 255, empty)
cv2.createTrackbar("Sat Max", "ColorRange", 255, 255, empty)
cv2.createTrackbar("Val Min", "ColorRange", 0, 255, empty)
cv2.createTrackbar("Val Max", "ColorRange", 255, 255, empty)


while(True) :
    ret1, rgb1 = cam1.read()

    if ret1 :
        hsv1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos("Hue Min", "ColorRange")
        h_max = cv2.getTrackbarPos("Hue Max", "ColorRange")
        s_min = cv2.getTrackbarPos("Sat Min", "ColorRange")
        s_max = cv2.getTrackbarPos("Sat Max", "ColorRange")
        v_min = cv2.getTrackbarPos("Val Min", "ColorRange")
        v_max = cv2.getTrackbarPos("Val Max", "ColorRange")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        mask1 = cv2.inRange(hsv1, lower, upper)

        cv2.imshow("Cam1", mask1)

    cv2.waitKey(5)

cv2.destroyAllWindows()