import cv2
import numpy as np
import pickle
import os

def empty(a) :
    pass

def save(path, object) :
    with open(path, "wb") as f :
        pickle.dump(object, f)

def load(path) :
    if not os.path.exists(path) :
        return ColorRange()
    with open(path, "rb") as f :
        return pickle.load(f)

        
class ColorRange :
    upper:np.ndarray
    lower:np.ndarray

    def __init__(self) :
        self.upper = np.array([179, 255, 255])
        self.lower = np.array([0, 0, 0])


    def getParameters(self, h_min, h_max, s_min, s_max, v_min, v_max) :
        h_min = cv2.getTrackbarPos("Hue Min", "cam")
        h_max = cv2.getTrackbarPos("Hue Max", "cam")
        s_min = cv2.getTrackbarPos("Sat Min", "cam")
        s_max = cv2.getTrackbarPos("Sat Max", "cam")
        v_min = cv2.getTrackbarPos("Val Min", "cam")
        v_max = cv2.getTrackbarPos("Val Max", "cam")

        self.upper = np.array([h_max, s_max, v_max])
        self.lower = np.array([h_min, s_min, v_min])


    def run(self, cam1, cam2) :
        cv2.namedWindow("cam")

        cv2.createTrackbar("Hue Min", "cam", self.lower[0], 179, empty)
        cv2.createTrackbar("Hue Max", "cam", self.upper[0], 179, empty)
        cv2.createTrackbar("Sat Min", "cam", self.lower[1], 255, empty)
        cv2.createTrackbar("Sat Max", "cam", self.upper[1], 255, empty)
        cv2.createTrackbar("Val Min", "cam", self.lower[2], 255, empty)
        cv2.createTrackbar("Val Max", "cam", self.upper[2], 255, empty)

        while True :
            ret1, frame1 = cam1.read()
            ret2, frame2 = cam2.read()

            if ret1 and ret2:
                self.getParameters(self.lower[0], self.upper[0], self.lower[1], self.upper[1], self.lower[2], self.upper[2])
                hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
                hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
                mask1 = cv2.inRange(hsv1, self.lower, self.upper)
                mask2 = cv2.inRange(hsv2, self.lower, self.upper)

                combined1 = cv2.hconcat([frame1, frame2])
                combined2 = cv2.hconcat([mask1, mask2])
                combined3 = cv2.vconcat([combined1, combined2])
                
                cv2.imshow("cam", combined3)
            else :
                break

            key = cv2.waitKey(5)
            if key == ord(" ") :
                break
            
        cv2.destroyAllWindows()



if __name__ == "__main__" :
    cam1 = cv2.VideoCapture(1)
    cam2 = cv2.VideoCapture(2)

    cr = load("color_range")
    
    cr.run(cam1, cam2)
    save("color_range", cr)