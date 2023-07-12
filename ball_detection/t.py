import cv2
import numpy as np

img = cv2.imread("b")
height = img.shape[0]
weight = img.shape[1]

src_point = np.float32([[50, 50], [100, 100], [70, 150], [0, 100]])
dst_point = np.float32([[50, 50], [100, 50], [100, 100], [50, 100]])

perspective_matrix = cv2.getPerspectiveTransform(src_point, dst_point)
warped_img = cv2.warpPerspective(img, perspective_matrix, (weight, height))

cv2.imshow("img", warped_img)
cv2.waitKey(0)