import cv2
import numpy as np
import pickle

class ColorMode:
    flag:str
    transformTo:int
    transformFrom:int
    maxValues:tuple
    names:tuple
    def __init__(self, flag, transformTo, transformFrom, maxValues, names):
        self.flag = flag
        self.transformTo = transformTo
        self.transformFrom = transformFrom
        self.maxValues = maxValues
        self.names = names

COLOR_MODES = {
    "RGB": ColorMode("RGB", cv2.COLOR_BGR2RGB, cv2.COLOR_RGB2BGR, (255, 255, 255), ("R", "G", "B")),
    "HSV": ColorMode("HSV", cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR, (179, 255, 255), ("H", "S", "V")),
    "LAB": ColorMode("LAB", cv2.COLOR_BGR2LAB, cv2.COLOR_LAB2BGR, (255, 255, 255), ("L", "A", "B")),
    "GRAY": ColorMode("GRAY", cv2.COLOR_BGR2GRAY, cv2.COLOR_GRAY2BGR, (255), ("GRAY"))
}

class ColorRange:
    mode:ColorMode
    lower:np.ndarray
    upper:np.ndarray
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def set(self, source, mode=COLOR_MODES["RGB"]):
        cv2.namedWindow("set range")
        self.mode = mode
        self.lower = np.zeros(len(self.mode.maxValues), np.uint8)
        self.upper = np.zeros(len(self.mode.maxValues), np.uint8)
        for i in range(len(self.mode.maxValues)):
            cv2.createTrackbar(self.mode.names[i] + " lower", "set range", 0, self.mode.maxValues[i], lambda x: self.lower.put(i, x))
            cv2.createTrackbar(self.mode.names[i] + " upper", "set range", 0, self.mode.maxValues[i], lambda x: self.upper.put(i, x))

        if isinstance(source, str):
            img = cv2.imread(source)
            def sowImage():
                img_ = cv2.cvtColor(img, self.mode.transformTo)
                thresholded = cv2.inRange(img_, self.lower, self.upper)
                cv2.imshow('set range', img_)
            cv2.createButton("upd", lambda x: sowImage())
            sowImage()
            cv2.waitKey(0)
        else:
            cap = cv2.VideoCapture(source)
            while True:
                ret, img = cap.read()
                img = cv2.cvtColor(img, self.mode.transformTo)
                thresholded = cv2.inRange(img, self.lower, self.upper)
                cv2.imshow('set range', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

if __name__ == "__main__":
    cr = ColorRange()
    cr.set("ball_sample.jpg", COLOR_MODES["HSV"])