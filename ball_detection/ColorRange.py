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


    def runColorRange_video(self, cam) :
        cv2.namedWindow("ColorRangeSetting")

        cv2.createTrackbar("Hue Min", "ColorRangeSetting", self.lower[0], 179, empty)
        cv2.createTrackbar("Hue Max", "ColorRangeSetting", self.upper[0], 179, empty)
        cv2.createTrackbar("Sat Min", "ColorRangeSetting", self.lower[1], 255, empty)
        cv2.createTrackbar("Sat Max", "ColorRangeSetting", self.upper[1], 255, empty)
        cv2.createTrackbar("Val Min", "ColorRangeSetting", self.lower[2], 255, empty)
        cv2.createTrackbar("Val Max", "ColorRangeSetting", self.upper[2], 255, empty)

        while True :
            ret, frame = cam.read()

            if ret :
                self.getParameters(self.lower[0], self.upper[0], self.lower[1], self.upper[1], self.lower[2], self.upper[2])

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, self.lower, self.upper)

                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                cv2.putText(mask, "Mask Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
                combined = cv2.hconcat([frame, mask])
                
                cv2.imshow("ColorRangeSetting", combined)
                #cv2.imshow("ColorRangeMask", combined)
            else :
                break

            key = cv2.waitKey(20)
            if key == ord(" ") :
                break
            
        cv2.destroyAllWindows()

    def runColorRange_image(self, img) :
        cv2.namedWindow("ColorRangeSetting")

        cv2.createTrackbar("Hue Min", "ColorRangeSetting", self.lower[0], 179, empty)
        cv2.createTrackbar("Hue Max", "ColorRangeSetting", self.upper[0], 179, empty)
        cv2.createTrackbar("Sat Min", "ColorRangeSetting", self.lower[1], 255, empty)
        cv2.createTrackbar("Sat Max", "ColorRangeSetting", self.upper[1], 255, empty)
        cv2.createTrackbar("Val Min", "ColorRangeSetting", self.lower[2], 255, empty)
        cv2.createTrackbar("Val Max", "ColorRangeSetting", self.upper[2], 255, empty)

        while True :    
            self.getParameters(self.lower[0], self.upper[0], self.lower[1], self.upper[1], self.lower[2], self.upper[2])

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower, self.upper)

            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(mask, "Mask Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, "Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            combined = cv2.hconcat([img, mask])

            cv2.imshow("ColorRangeSetting", combined)
            if cv2.waitKey(1) == ord(" ") :
                break



if __name__ == "__main__" :
    cam = cv2.VideoCapture("ball_detection/result/20230718-2/all.mp4")
    #img = cv2.imread("ball_sample.jpg")

    cr = load("color_range")
    
    cr.runColorRange_video(cam)
    #cr.runColorRange_image(img)
    save("color_range", cr)