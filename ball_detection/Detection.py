import cv2
import numpy as np
import pickle
import os
import math
import threading
import multiprocessing as mp
from ColorRange import *
from pupil_apriltags import Detector


def homography_matrix(source) :
    tag_len = 12.9 #set tag length (cm)
    detector = Detector()
    detection = detector.detect(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY))
    coners = detection[0].corners
    tar = np.float32([[0, tag_len], [tag_len, tag_len], [tag_len, 0], [0, 0]])
    homography = cv2.findHomography(coners, tar)[0]
    return homography


class Detection :
    #colorrange
    range = load("color_range")
    upper = range.upper
    lower = range.lower

    #save test frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    def __init__(self, frame_size=(640,480), frame_rate=30, rangeFile="color_range", save_name=None) :
        self.frame_size = frame_size
        self.frame_rate = frame_rate
        self.range = load(rangeFile)
        self.upper = self.range.upper
        self.lower = self.range.lower


    def writeVideo(self, path, frame) :
        video_writer = cv2.VideoWriter(path, self.fourcc, self.frame_rate, self.frame_size)
        video_writer.write(frame)

    def drawDirection(self, frame, x, y, h, w) :
        xCenter = x + w // 2
        yCenter = y + h // 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (xCenter, yCenter), 2, (0, 255, 0), -1)
        #cv2.putText(frame, ("x : {} y : {}".format(xCenter, yCenter)), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

    def compareFrames(self, frame, compare) :
        move = cv2.bitwise_xor(frame, compare)
        color = cv2.inRange(move, np.array([28, 28, 20]), np.array([255, 255, 255]))
        return cv2.bitwise_and(frame, cv2.cvtColor(color, cv2.COLOR_GRAY2BGR))
    
    def maskFrames(self, frame) :
        return cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), self.lower, self.upper)

    def detectContours(self, frame) :
        return cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

    def ballFeature(self,area, h, w) :
        r = math.sqrt(h*h + w*w) / 2
        a = math.pi * r * r
        rmin = 3.7178529388965496/2.0
        rmax = 18.589264694482747/2.0
        circle = area / a
        if not rmin < r < rmax:
            return False
        if not math.pi * rmin * rmin < area < math.pi * rmax * rmax:
            return False
        if not 0 < circle < 100 :
            return False
        
        return True
   
 
    def runDetevtion(self, apriltag_source, source, save_name) :
        cam = cv2.VideoCapture(source)
        whetherTheFirstFrame = True

        while(True) :
            numberOfBall = 0
            ret, frame = cam.read()

            if ret :
                if whetherTheFirstFrame :
                    compare = frame
                    whetherTheFirstFrame = False
                    continue
                
                for contour in self.detectContours(self.maskFrames(self.compareFrames(frame, compare))) :
                    area = cv2.contourArea(contour)
                    x, y, w, h = cv2.boundingRect(contour)
                    if self.ballFeature(area, h, w) :
                        self.drawDirection(frame, x, y, h, w)
                        numberOfBall += 1
                        ball_in_world = np.dot(homography_matrix(apriltag_source), np.array([frame.shape[0] - (x+w//2), y+h//2, 1]))
                        print("({}, {})".format(ball_in_world[0], ball_in_world[1]))

                if not numberOfBall == 1 :
                    self.writeVideo("ball_detection/TestVideos/" + save_name + "_bad.mp4", frame)

                self.writeVideo("ball_detection/TestVideos/" + save_name + "_tagged.mp4", frame)
                
                window = "Camera " + str(source) 
                cv2.imshow(window, frame)

                if cv2.waitKey(100) == ord(' ') :
                    break

        

if __name__ == "__main__" :
    detector1 = Detection()
    detector2 = Detection()

    camera1 = mp.Process(target=detector1.runDetevtion, args=(0, "apriltag_source", "bad_1.mp4", "all_1.mp4"))
    camera2 = mp.Process(target=detector2.runDetevtion, args=(1, "apriltag_source", "bad_2.mp4", "all_2.mp4"))
    
    camera1.start()
    camera2.start()
    

    cv2.destroyAllWindows()

    #img = cv2.imread("ball_detection/apriltag-pad.jpg")
    #result = ketstone_correction("ball_detection/apriltag-pad.jpg")
    #src_point = np.float32([[result[0], result[1]], [result[2], result[3]], [result[4], result[5]], [result[6], result[7]]])
    #dst_point = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])
    #perspective_matrix = cv2.getPerspectiveTransform(src_point, dst_point)
    #warped = cv2.warpPerspective(img, perspective_matrix, (640, 480))
    #cv2.imshow("warped", warped)
    #cv2.waitKey(0)