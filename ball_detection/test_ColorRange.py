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
        h_min = cv2.getTrackbarPos("Hue Min", "ColorRangeSetting")
        h_max = cv2.getTrackbarPos("Hue Max", "ColorRangeSetting")
        s_min = cv2.getTrackbarPos("Sat Min", "ColorRangeSetting")
        s_max = cv2.getTrackbarPos("Sat Max", "ColorRangeSetting")
        v_min = cv2.getTrackbarPos("Val Min", "ColorRangeSetting")
        v_max = cv2.getTrackbarPos("Val Max", "ColorRangeSetting")

        self.upper = np.array([h_max, s_max, v_max])
        self.lower = np.array([h_min, s_min, v_min])


    def run(self, cam1) :
        cv2.namedWindow("ColorRangeSetting")

        cv2.createTrackbar("Hue Min", "ColorRangeSetting", self.lower[0], 179, empty)
        cv2.createTrackbar("Hue Max", "ColorRangeSetting", self.upper[0], 179, empty)
        cv2.createTrackbar("Sat Min", "ColorRangeSetting", self.lower[1], 255, empty)
        cv2.createTrackbar("Sat Max", "ColorRangeSetting", self.upper[1], 255, empty)
        cv2.createTrackbar("Val Min", "ColorRangeSetting", self.lower[2], 255, empty)
        cv2.createTrackbar("Val Max", "ColorRangeSetting", self.upper[2], 255, empty)

        while True :
            ret1, frame1 = cam1.read()

            if ret1 :
                self.getParameters(self.lower[0], self.upper[0], self.lower[1], self.upper[1], self.lower[2], self.upper[2])

                hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
                mask1 = cv2.inRange(hsv1, self.lower, self.upper)

                cv2.putText(frame1, "Camera 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)
                cv2.putText(mask1, "Mask Camera 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 239, 97), 2)
            
                
                cv2.imshow("ColorRangeSetting", frame1)
                cv2.imshow("ColorRangeMask", mask1)
            else :
                break

            key = cv2.waitKey(50)
            if key == ord(" ") :
                break
            
        cv2.destroyAllWindows()



if __name__ == "__main__" :
    cam1 = cv2.VideoCapture('test.mp4')

    cr = load("test_color_range")
    
    cr.run(cam1)
    save("test_color_range", cr)