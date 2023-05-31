from pupil_apriltags import Detector, Detection
import cv2
import time
import pickle


with open("calibration1", "rb") as f:
    cameraMatrix= pickle.load(f)

def f1():
    cap = cv2.VideoCapture(0)
    a = False
    while True :
        ret, frame = cap.read()
        if ret :
            cv2.imshow('frame', frame)
            key = cv2.waitKey(10)

            if a :
                cv2.imwrite('test.jpg', frame)
                break

            if key == ord('q') :
                #sleep for one second
                #time.sleep(5)
                a = True

def f2() :
    cap = cv2.VideoCapture(0)
    a = False
    while True :
        ret, frame = cap.read()
        time.sleep(1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #load the detector
        detector = Detector()

        results = detector.detect(image, estimate_tag_pose=True, camera_params=(cameraMatrix[0][0],cameraMatrix[1][1],cameraMatrix[0][2],cameraMatrix[1][2]), tag_size=11)
        
        if len(results) == 1:
            res:Detection = results[0]
            print(res.pose_t[0][0], res.pose_t[1][0], res.pose_t[2][0])
        else:
            print("no tag detected")

def getBallPixelSize(distance) :
    BALL_REAL_SIZE = 3.8
    return BALL_REAL_SIZE * cameraMatrix[0][0] / distance

print(getBallPixelSize(100), getBallPixelSize(450))