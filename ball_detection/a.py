import cv2

# 创建视频捕获对象
cap = cv2.VideoCapture(2)  # 使用摄像头索引0（通常为内置摄像头）

# 设置视频编码器和输出参数（可选）
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用XVID编码器
out = cv2.VideoWriter('output.avi', fourcc, 60.0, (1280, 720))
# 参数解释：
# - 'output.avi'：输出文件的名称
# - fourcc：编码器
# - 20.0：帧率（每秒20帧）
# - (640, 480)：帧的大小（宽度和高度）

while True:
    ret, frame = cap.read()  # 读取一帧
    if not ret:
        break  # 如果无法读取帧，退出循环

    # 在这里可以对帧进行任何处理（例如，添加文本或绘图）

    # 将帧写入输出视频（如果使用了视频写入对象）
    out.write(frame)

    # 显示帧（可选）
    cv2.imshow('Frame', frame)

    # 要停止录制，请按 'q' 键
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()

# 如果使用了视频写入对象，也要释放它
out.release()

# 关闭所有窗口
cv2.destroyAllWindows()

