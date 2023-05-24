import cv2

a = cv2.imread("ball_detection/test1.png")
b = cv2.imread("ball_detection/test1.png")
a = cv2.cvtColor(a, cv2.COLOR_BGR2HSV_FULL)

cv2.imshow("aa", cv2.bitwise_xor(a, b))
cv2.waitKey(0)