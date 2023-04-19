import cv2
import numpy as np

def empty(a) :
    pass


img = cv2.imread("ball_sample.jpg")
img2 = cv2.imread("ball_sample.jpg")
combined1 = cv2.hconcat([img, img2])
combined2 = cv2.hconcat([img, img2])
combined3 = cv2.vconcat([combined1, combined2])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.namedWindow("img")

cv2.createTrackbar("Hue Min", "img", 0, 179, empty)
cv2.createTrackbar("Hue Max", "img", 179, 179, empty)
cv2.createTrackbar("Sat Min", "img", 0, 255, empty)
cv2.createTrackbar("Sat Max", "img", 255, 255, empty)
cv2.createTrackbar("Val Min", "img", 0, 255, empty)
cv2.createTrackbar("Val Max", "img", 255, 255, empty)

while True :
    h_min = cv2.getTrackbarPos("Hue Min", "img")
    h_max = cv2.getTrackbarPos("Hue Max", "img")
    s_min = cv2.getTrackbarPos("Sat Min", "img")
    s_max = cv2.getTrackbarPos("Sat Max", "img")
    v_min = cv2.getTrackbarPos("Val Min", "img")
    v_max = cv2.getTrackbarPos("Val Max", "img")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower, upper)
    cv2.imshow("img", combined3)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) == ord(' ') :
        break