import cv2
import numpy as np

# 准备棋盘格角点的标定板尺寸
pattern_size = (9, 6)  # 内部角点数量
square_size = 1.0  # 棋盘格方块的大小（单位：任意）

# 准备用于存储图像和角点的数组
obj_points = []  # 3D 空间点
img_points = []  # 2D 图像点

# 检测角点并构建角点坐标
def find_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        obj_points.append(np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32))
        obj_points[-1][:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        img_points.append(corners)

# 从摄像头读取图像并进行标定
def calibrate_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        find_corners(frame)
        cv2.drawChessboardCorners(frame, pattern_size, img_points[-1], True)
        cv2.imshow("Chessboard", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            # 标定相机
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
            print("相机内参（相机矩阵）:\n", mtx)
            print("畸变系数:\n", dist)
            break

    cap.release()
    cv2.destroyAllWindows()

# 运行相机标定
calibrate_camera()
