import cv2
import numpy as np
import pickle
import glob
import os

#def save

class Calibrator():
    def __init__ (self) :
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.object_points = np.zeros((6*9,3), np.float32)
        self.object_points[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) * 2.3
        self.points2D = []
        self.points3D = []

    def runFindCorners(self, frame, gray) :
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        if ret :
            self.points3D.append(self.object_points)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), self.criteria)
            self.points2D.append(corners2)
            cv2.drawChessboardCorners(frame, (9,6), corners2, ret)
        self.size = gray.shape[::-1]
    
    def runCalibrate(self) :
        #print("calibrating...") 
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.points3D, self.points2D, self.size, None, None)
        #print("ret: {}".format(ret))
        print("intrinsic matrix: \n {}".format(mtx))
        # in the form of (k_1, k_2, p_1, p_2, k_3)
        #print("distortion cofficients: \n {}".format(dist))
        #print("rotation vectors: \n {}".format(rvecs))
        #print("translation vectors: \n {}".format(tvecs))
    
    def run(self, source) :
        countFrame = 0
        cam = cv2.VideoCapture(source)
        while True :
            ret, frame = cam.read()
            print(frame.shape)
            countFrame += 1
            if not countFrame % 30  :
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.runFindCorners(frame, gray)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('c') :
                cv2.destroyAllWindows()
                self.runCalibrate()
                break






if __name__ == ("__main__") :
    calibrate = Calibrator()
    calibrate.run(0)



