import cv2
import numpy as np
import pickle
import os
import sys
sys.path.append(os.getcwd())
from camera_reciever.CameraReceiver import CameraReceiver

def save(path, object) :
    with open(path, "wb") as f :
        pickle.dump(object, f)

def load(path) :
    if not os.path.exists(path) :
        return ColorRange()
    with open(path, "rb") as f :
        return pickle.load(f)

        
hsv:np.ndarray = None
def on_mouse(event, x, y, flags, param):
    if hsv is not None :
        if event == cv2.EVENT_LBUTTONDOWN:
            print("H:", hsv[y, x, 0], " S:", hsv[y, x, 1], " V:", hsv[y, x, 2])

class ColorRange :
    upper:np.ndarray
    lower:np.ndarray

    def __init__(self, upper = [179, 255, 255], lower = [0, 0, 0]) :
        self.upper = np.array(upper)
        self.lower = np.array(lower)


    def getParameters(self, h_min, h_max, s_min, s_max, v_min, v_max) :
        h_min = cv2.getTrackbarPos("Hue Min", "ColorRangeSetting")
        h_max = cv2.getTrackbarPos("Hue Max", "ColorRangeSetting")
        s_min = cv2.getTrackbarPos("Sat Min", "ColorRangeSetting")
        s_max = cv2.getTrackbarPos("Sat Max", "ColorRangeSetting")
        v_min = cv2.getTrackbarPos("Val Min", "ColorRangeSetting")
        v_max = cv2.getTrackbarPos("Val Max", "ColorRangeSetting")

        self.upper = np.array([h_max, s_max, v_max])
        self.lower = np.array([h_min, s_min, v_min])


    def runColorRange_video(self, source, recursive = False, save_file_name = None, f2f = False) :
        cv2.namedWindow("ColorRangeSetting")
        global hsv
        cv2.setMouseCallback("ColorRangeSetting", on_mouse)

        cv2.createTrackbar("Hue Min", "ColorRangeSetting", self.lower[0], 179, empty)
        cv2.createTrackbar("Hue Max", "ColorRangeSetting", self.upper[0], 179, empty)
        cv2.createTrackbar("Sat Min", "ColorRangeSetting", self.lower[1], 255, empty)
        cv2.createTrackbar("Sat Max", "ColorRangeSetting", self.upper[1], 255, empty)
        cv2.createTrackbar("Val Min", "ColorRangeSetting", self.lower[2], 255, empty)
        cv2.createTrackbar("Val Max", "ColorRangeSetting", self.upper[2], 255, empty)


        if type(source) == str and source.replace(".", "").isdigit() :
            cam = CameraReceiver(source)
            cam.connect()
        else :
            cam = cv2.VideoCapture(source)
        i = 0
        read = True
        while True :
            i += 1
            if read :
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
                if recursive :
                    cam = cv2.VideoCapture(source)
                else :
                    break

            if i % 30 == 0 :
                if save_file_name is not None :
                    save(save_file_name, self)
            # wait to match the fps of the video
            key = cv2.waitKey(round(1/30 * 1000) if not f2f else 1)
            if key == ord(" ") :
                break
            elif f2f :
                if key == ord("n") :
                    read = True
                else :
                    read = False
        
        if type(source) == str and source.replace(".", "").isdigit() :
            cam.close()
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

cr = None

def empty(a) :
    if cr is not None:
        save("color_range_1", cr)

if __name__ == "__main__" :
    cr = load("cr3")
    cr.runColorRange_video("exp/3.mp4", recursive=True, f2f=True)
    save("cr3", cr)
    #cr.runColorRange_video("ball_detection/result/hd_60_detection_r2/bad.mp4", recursive=True)