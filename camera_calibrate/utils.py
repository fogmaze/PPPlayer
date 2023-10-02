import numpy as np
from pupil_apriltags import Detector, Detection
import sys
import os
sys.path.append(os.getcwd())
import core.Equation3d as equ
import camera_calibrate.Calibration as calib
from core.Constants import *
import cv2
import csv
import time
import pickle
from camera_reciever.CameraReceiver import CameraReceiver


def takePicture_and():
    t = time.time()
    cap = CameraReceiver("172.20.10.2")
    #lprint(cap.get(cv2.CAP_PROP_FPS))
    a = False
    i = 0
    cap.connect()
    while True :
        ret, frame = cap.read()
        #print(1/(time.time() - t))
        t = time.time()
        if ret :
            cv2.imshow('frame', frame)
            key = cv2.waitKey(10) & 0xff

            if a :
                cv2.imwrite('pic0.jpg', frame)
                break

            if key == ord('w') :
                cv2.imwrite('D{}.jpg'.format(i), frame)
                i += 1


            if key == ord('q') :
                #sleep for one second
                #time.sleep(5)
                a = True
    cap.close()
    cv2.destroyAllWindows()

def takePicture():
    t = time.time()
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FPS, 60)
    #lprint(cap.get(cv2.CAP_PROP_FPS))
    a = False
    i = 0
    while True :
        ret, frame = cap.read()
        #print(1/(time.time() - t))
        t = time.time()
        if ret :
            cv2.imshow('frame', frame)
            key = cv2.waitKey(10) & 0xff

            if a :
                cv2.imwrite('B.jpg', frame)
                break

            if key == ord('w') :
                cv2.imwrite('D{}.jpg'.format(i), frame)
                i += 1


            if key == ord('q') :
                #sleep for one second
                #time.sleep(5)
                a = True
    cap.release()
    cv2.destroyAllWindows()

def calculateCameraPosition(cameraMatrix:np.ndarray, frame_gray, tagSize=APRILTAG_SIZE) :
    try :
        detector = Detector()
        results = detector.detect(frame_gray, 
                                    estimate_tag_pose=True, 
                                    camera_params=(cameraMatrix[0][0],cameraMatrix[1][1],cameraMatrix[0][2],cameraMatrix[1][2]),
                                    tag_size=tagSize)
        if len(results) == 1:
            res:Detection = results[0]
            position = np.matmul(np.linalg.inv(res.pose_R), -res.pose_t)
            return equ.Point3d(position[0][0] + (tagSize/2) - 2.74/2, -position[2][0] - 1.525/2, -position[1][0] + (tagSize/2))
        else:
            print(1)
            return None
    except Exception as e:
        print(e)
        return None
    

def getCameraPosition_realTime(cameraMatrix) :
    cap = cv2.VideoCapture(0)
    while True :
        ret, frame = cap.read()
        time.sleep(1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        res = calculateCameraPosition(cameraMatrix, image)

        if res is not None :
            print(res.to_str())
        
def getBallPixelSize(distance, cameraMatrix) :
    BALL_REAL_SIZE = 0.038
    return BALL_REAL_SIZE * cameraMatrix[0][0] / distance


def runExerment() :
    cameraMatrix = pickle.load(open('calibration1_old', 'rb'))
    with open('experiment_AprilTag/data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for c in ('A', 'B', 'C', 'D') :
            res_x = []
            res_y = []
            res_z = []
            for i in range(10) :
                img = cv2.imread('experiment_AprilTag/{}/{}{}.jpg'.format(c, c, i), cv2.IMREAD_GRAYSCALE)
                pos = calculateCameraPosition(cameraMatrix, img)
                res_x.append(pos.x)
                res_y.append(pos.y)
                res_z.append(pos.z)
                print(pos.to_str())
            writer.writerow(res_x)
            writer.writerow(res_y)
            writer.writerow(res_z)
            
if __name__ == "__main__" :
    takePicture_and()
    exit()
    #runExerment()
    cameraMatrix = pickle.load(open('calibration', 'rb'))
    #takePicture()
    img = cv2.imread("718-2.jpg", cv2.IMREAD_GRAYSCALE)
    #dec = Detector()
    #d = dec.detect(img, estimate_tag_pose=True, camera_params=(cameraMatrix[0][0],cameraMatrix[1][1],cameraMatrix[0][2],cameraMatrix[1][2]), tag_size=29.8)
    #cv2.imshow('frame', img)
    #cv2.waitKey(0)
    
    print(calculateCameraPosition(cameraMatrix, img).to_str())


    #getCameraPosition_realTime(cameraMatrix)
    #calculateCameraPosition(cameraMatrix, cv2.imread('test.jpg'))
    #print(cameraMatrix)
    #getCameraPosition_realTime(cameraMatrix)
    pass