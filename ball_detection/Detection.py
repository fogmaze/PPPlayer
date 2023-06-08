import cv2
import numpy as np
import pickle
import os
import math
from ColorRange import *



class Detection :
    range = load("color_range")
    upper = range.upper
    lower = range.lower

    def drawDirection(self, frame, x, y, h, w) :
        xCenter = x + w // 2
        yCenter = y + h // 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (xCenter, yCenter), 2, (0, 255, 0), -1)
        cv2.putText(frame, ("x : {} y : {}".format(xCenter, yCenter)), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

    def compareFrames(self, frame, compare) :
        move = cv2.bitwise_xor(frame, compare)
        color = cv2.inRange(move, np.array([28, 28, 20]), np.array([255, 255, 255]))
        return cv2.bitwise_and(frame, cv2.cvtColor(color, cv2.COLOR_GRAY2BGR))
    
    def maskFrames(self, frame) :
        return cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), self.lower, self.upper)

    def detectContours(self, frame) :
        return cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

    # value hasn't been measured
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
 
    def runDetevtion(self, cam1, cam2) :
        whetherTheFirstFrame = True

        while(True) :
            ret1, frame1 = cam1.read()
            ret2, frame2 = cam2.read()

            if ret1 and ret2 :
                if whetherTheFirstFrame :
                    compare1 = frame1
                    compare2 = frame2
                    whetherTheFirstFrame = False
                    continue
                
                # area hasn't been measured
                for contour in self.detectContours(self.maskFrames(self.compareFrames(frame1, compare1))) :
                    area = cv2.contourArea(contour)
                    print(area)
                    if area > 0:
                        x, y, w, h = cv2.boundingRect(contour)
                        if self.ballFeature(area, h, w) :
                            self.drawDirection(frame1, x, y, h, w)
                for contour in self.detectContours(self.maskFrames(self.compareFrames(frame2, compare2))) :
                    area = cv2.contourArea(contour)
                    if area > 0 :
                        x, y, w, h = cv2.boundingRect(contour)
                        if self.ballFeature(area, h, w) :
                            self.drawDirection(frame2, x, y, h, w)

                cv2.imshow("Camera 1", frame1)
                cv2.imshow("Camera 2", frame2)


                if cv2.waitKey(100) == ord(' ') :
                    break

        

if __name__ == "__main__" :
    cam1 = cv2.VideoCapture('output.mp4')
    cam2 = cv2.VideoCapture(0)

   # output_file = 'output.mp4'
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编解码器
    #frame_rate = 30.0  # 帧率为30fps
    #frame_size = (640, 480)  # 帧尺寸为640x480

    #video_writer = cv2.VideoWriter(output_file, fourcc, frame_rate, frame_size)

    #while True :
        #ret, frame = cam1.read()
        #if ret :
            #video_writer.write(frame)
            #cv2.imshow("Camera 1", frame)
            #if cv2.waitKey(20) == ord(' ') :
                #break

    detector = Detection()
    detector.runDetevtion(cam1, cam2)



    cv2.destroyAllWindows()
    cam1.release()
    cam2.release()
