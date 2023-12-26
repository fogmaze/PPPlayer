import time
from typing import Tuple
import shutil
import cv2
import numpy as np
import math
import multiprocessing as mp
from pupil_apriltags import Detector
import csv
import sys
import os
sys.path.append(os.getcwd())
from ball_detection.ColorRange import *
import core.display as display
import core.common as common
from camera_calibrate.utils import *
from core.Constants import *
import core.Constants as Constants
import core.Equation3d as equ
import camera_calibrate.utils as utils
import camera_calibrate.Calibration as calib
import argparse as ap
from camera_reciever.CameraReceiver import CameraReceiver
import time

# this function is no longer used.
def find_homography_matrix_to_apriltag(img_gray) -> np.ndarray | None:
    tag_len   = APRILTAG_SIZE 
    detector  = Detector()
    detection = detector.detect(img_gray)
    if len(detection) == 0 :
        return None
    coners = detection[0].corners
    tar    = np.float32([[0, 0], [tag_len, 0], [tag_len, tag_len], [0, tag_len]])
    homography = cv2.findHomography(coners, tar)[0]
    return homography

# check if ball bounced. (used to sync two cameras)
class _bounce_checker :
    def __init__(self):
        self.movement = None
        self.last_y = None
        self.this_frame_is_bounce = False
    
    # update the bounce checker with the y position of the ball. return True if bounce detected.
    def update(self, y) :
        if self.movement == 1:
            if y < self.last_y:
                self.movement = None
                self.last_y = None
                self.this_frame_is_bounce = True
                return True
        elif self.movement == -1 :
            if y > self.last_y:
                self.movement = None
                self.last_y = None
                self.this_frame_is_bounce = True
                return True
        elif self.movement == None and self.last_y is not None:
            self.movement = 1 if y - self.last_y > 0 else -1
            self.this_frame_is_bounce = False
        self.last_y = y
        return False

# used to store configuration for one detection. ex: frame size, frame rate, color range, etc.
class DetectionConfig :
    def __init__(self) :
        self.frame_rate = 30
        self.frame_size = (640, 480)
        self.camera_position = None
        self.homography_matrix = None
        self.range = None
        self.consider_poly = None
        self.camera_info:TableInfo = None
        self.upper = None
        self.lower = None
        self.configFileName = None

    def load(self, config) :
        self.configFileName = config
        if not os.path.exists(os.path.join("configs", config)) :
            raise Exception("config dir not found")
        if os.path.exists(os.path.join("configs", config, "frame_size")) :
            self.frame_size = load(os.path.join("configs", config, "frame_size"))
        if os.path.exists(os.path.join("configs", config, "frame_rate")) :
            self.frame_rate = load(os.path.join("configs", config, "frame_rate"))
        if os.path.exists(os.path.join("configs", config, "color_range")) :
            self.range = load(os.path.join("configs", config, "color_range"))
        else :
            raise Exception("color range file not found")
        if os.path.exists(os.path.join("configs", config, "consider_poly")) :
            self.consider_poly = load(os.path.join("configs", config, "consider_poly"))
        if os.path.exists(os.path.join("configs", config, "camera_info")) :
            self.camera_info = load(os.path.join("configs", config, "camera_info"))

        if self.consider_poly is None :
            self.consider_poly = np.array([[0, 0], [self.frame_size[0], 0], [self.frame_size[0], self.frame_size[1]], [0, self.frame_size[1]]])

        self.upper = self.range.upper
        self.lower = self.range.lower

        if self.camera_info is not None :
            self.camera_position = calculateCameraPosition_table(self.camera_info)
            self.inmtx = self.camera_info.inmtx
            w =  self.camera_info.width
            h =  self.camera_info.height
            self.homography_matrix = cv2.findHomography(self.camera_info.corners, np.float32([[-w/2, h/2], [w/2, h/2], [w/2, -h/2], [-w/2, -h/2]]))[0]

    def save(self, save_name) :
        if self.camera_info is not None :
            with open("configs/" + save_name + "/camera_info", "wb") as f :
                pickle.dump(self.camera_info, f)
        if self.range is not None :
            with open("configs/" + save_name + "/color_range", "wb") as f :
                pickle.dump(self.range, f)
        if self.frame_rate is not None :
            with open("configs/" + save_name + "/frame_rate", "wb") as f :
                pickle.dump(self.frame_rate, f)
        if self.frame_size is not None :
            with open("configs/" + save_name + "/frame_size", "wb") as f :
                pickle.dump(self.frame_size, f)
        if self.consider_poly is not None :
            with open("configs/" + save_name + "/consider_poly", "wb") as f :
                pickle.dump(self.consider_poly, f)

def createConfig(source, save_name, frame_size=(640,480), frame_rate=30, color_range="cr3", camera_info=None, consider_poly=None, ini_img=None) :
    common.replaceDir("configs", save_name)
    config = DetectionConfig()
    
    if camera_info is not None:
        config.camera_info = camera_info

    if consider_poly is not None :
        config.consider_poly = consider_poly
    else :
        poly = setup_poly(source)
        config.consider_poly = poly
    if os.path.exists(os.path.join("configs", color_range)) :
        config.range = load(os.path.join("configs", color_range))
    else :
        raise Exception("color range file not found")
    config.frame_rate = frame_rate
    config.frame_size = frame_size
    config.save(save_name)


class Detection :
    #save test frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    MEAN_BALL_SIZE_DICT = {
        "320" : 175.5024938405144,
        "720" : 536.6738865832444,
        "1080" : 1207.5162448113356,
    }

    def __init__(self, 
                source, 
                mode,
                frame_size             = None, 
                frame_rate             = None, 
                color_range            = None, 
                save_name              = "default", 
                queue                  = None,
                conn                   = None,
                config:DetectionConfig = None,
                camera_info            = None,
                consider_poly          = None,
                ) :

        # mode for different uses of detection. 
        # "dual_analysis" for predict, which will send data to main process and save data into files
        # "analysis"      for single camera detection, which will save data into files
        # "compute"       for single camera detection, which will save data into self.data. And can be used for other operatio.
        self.mode = mode

        # below is all the initalization for all the modes
        self.last_map = []
        self.last_result = None

        print("source: ",source)
        self.save_name = save_name
        
        self.frame_rate = 30 
        self.frame_size = (640, 480)
        self.camera_position = None
        self.homography_matrix = None
        self.range = None
        self.consider_poly = None
        
        if config is not None :
            self.frame_rate = config.frame_rate
            self.frame_size = config.frame_size
            self.camera_position = config.camera_position
            self.homography_matrix = config.homography_matrix
            self.range = config.range
            self.consider_poly = config.consider_poly

        self.pid = os.getpid()

        self.frame_size = frame_size if frame_size is not None else self.frame_size
        self.frame_rate = frame_rate if frame_rate is not None else self.frame_rate
        self.consider_poly = consider_poly if consider_poly is not None else self.consider_poly
        if self.consider_poly is None :
            self.consider_poly = np.array([[0, 0], [self.frame_size[0], 0], [self.frame_size[0], self.frame_size[1]], [0, self.frame_size[1]]])
        if type(color_range) == str :
            self.range = load(color_range)
        elif type(color_range) == ColorRange :
            self.range = color_range
        self.upper = self.range.upper
        self.lower = self.range.lower
        self.video_writer_all = None
        self.video_writer_bad = None
        self.video_writer_tagged = None
        self.cam = None
        self.source = source
        self.data = []
        self.conn = None
        if self.frame_size == (640,480) :
            self.meanBallSize = self.MEAN_BALL_SIZE_DICT["320"]
        elif self.frame_size == (1920,1080) :
            self.meanBallSize = self.MEAN_BALL_SIZE_DICT["1080"]
        elif self.frame_size == (1280,720) :
            self.meanBallSize = self.MEAN_BALL_SIZE_DICT["720"] 
        else :
            raise Exception("frame size is not supported")

        if type(source) == str or type(source) == int:
            #self.cam = cv2.VideoCapture(source)
            pass

        if camera_info is not None:
            self.camera_position = calculateCameraPosition_table(camera_info)
            w =  camera_info.width
            h =  camera_info.height
            self.homography_matrix = cv2.findHomography(camera_info.corners, np.float32([[-w/2, h/2], [w/2, h/2], [w/2, -h/2], [-w/2, -h/2]]))[0]

        if self.mode == "analysis" or self.mode == "dual_analysis":
            common.replaceDir("results", save_name)
            self.video_writer_all = cv2.VideoWriter("results/" + save_name + "/all.mp4", self.fourcc, self.frame_rate, self.frame_size)
            self.video_writer_bad = cv2.VideoWriter("results/" + save_name + "/bad.mp4", self.fourcc, self.frame_rate, self.frame_size)
            self.video_writer_tagged = cv2.VideoWriter("results/" + save_name + "/tagged.mp4", self.fourcc, self.frame_rate, self.frame_size)
            self.video_writer_all_tagged = cv2.VideoWriter("results/" + save_name + "/all_tagged.mp4", self.fourcc, self.frame_rate, self.frame_size)
            self.detection_csv = open("results/" + save_name + "/detection.csv", "w", newline='')
            self.detection_csv_writer = csv.writer(self.detection_csv)
            self.detection_csv_writer.writerow(["iter", "id", "x", "y", "h", "w", "cam_x", "cam_y", "cam_z","rxy", "rxz"])
            # pickle camera position and homography matrix
            if config.configFileName is not None if config is not None else False:
                with open("results/" + save_name + "/config", "w") as f :
                    f.write(config.configFileName)
        if self.mode == "dual_analysis" or self.mode == "dual_run":
            if queue is None or self.camera_position is None or self.homography_matrix is None:
                raise Exception("dual_analysis mode need pipe, cam_pos and homography_matrix but {} {} {}".format(queue, self.camera_position, self.homography_matrix))
            self.bounce_checker = _bounce_checker()
            self.queue = queue
            self.conn = conn
        if self.mode =="caculate_bounce":
            self.bounce_checker = _bounce_checker()
            self.conn = conn
            
    def __del__(self) :
        if self.mode == "analysis" or self.mode == "dual_analysis":
            if self.video_writer_all is not None :
                self.video_writer_all.release()
            if self.video_writer_bad is not None :
                self.video_writer_bad.release()
            if self.video_writer_tagged is not None :
                self.video_writer_tagged.release()
            if self.video_writer_all_tagged is not None :
                self.video_writer_all_tagged.release()
            if self.detection_csv is not None :
                self.detection_csv.close()
        try :
            cv2.destroyAllWindows()
        except :
            pass
    
            
    # draw a retangle directly on the frame. not returning anything
    def drawDirection(self, frame, x, y, h, w, i) :
        xCenter = x + w // 2
        yCenter = y + h // 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (xCenter, yCenter), 2, (0, 0, 255), -1)
        cv2.putText(frame, ("x : {} y : {}".format(xCenter, yCenter)), (10, 40*i), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

    # this function is no longer used.
    def compareFrames(self, frame, compare) :
        move = cv2.bitwise_xor(frame, compare)
        color = cv2.inRange(move, np.array([10, 10, 10]), np.array([255, 255, 255]))
        return cv2.bitwise_and(frame, cv2.cvtColor(color, cv2.COLOR_GRAY2BGR))
    
    # this function is no longer used.
    def isBallFeature(self,area, h, w) :
        return True
        r = math.sqrt(h*h + w*w) / 2
        a = math.pi * r * r
        rmin = 3.7178529388965496/2.0
        rmax = 80.589264694482747/2.0
        circle = area / a
        if not rmin < r < rmax:
            return False
        if not math.pi * rmin * rmin < area < math.pi * rmax * rmax:
            return False
        if not 0.7 < circle < 1 :
            return False
        return True

    # get the next frame from the camera
    def getNextFrame(self) :
        return self.cam.read()
    
    # return a retangle that is considered as the ball
    def findBallInFrame(self, frame) -> Tuple[int, int, int, int] | None:
        masked = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), self.lower, self.upper)

        detected_contours = cv2.findContours(masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

        #cv2.polylines(frame, [self.consider_poly], True, (0, 255, 0), 2)
        
        detected_rects = []
        for contour in detected_contours :
            x, y, w, h = cv2.boundingRect(contour)
            detected_rects.append((x, y, w, h))

        merged = merge_rectangles(detected_rects)

        f0 = []
        for x, y, w, h in merged :
            if cv2.pointPolygonTest(self.consider_poly, (x+w//2, y+h//2), False) >= 0 :
                f0.append((x, y, w, h))
        f1 = []

        for x, y, w, h in f0:
            q = True
            for l in self.last_map:
                for x1, y1, w1, h1 in l :
                    if abs(x - x1) < 5 and abs(y - y1) < 5 :
                        q = False
                        break
            if q :
                f1.append((x, y, w, h))


        f2 = []
        for x, y, w, h in f1 :
            if self.meanBallSize * 0.05 < w * h < self.meanBallSize * 2:
                f2.append((x, y, w, h))
        result = None
        if self.last_result is not None :
            min_dist = 10000000
            for x, y, w, h in f2 :
                dis = math.sqrt(abs(x - self.last_result[0])**2 + abs(y - self.last_result[1])**2)
                if dis < min_dist:
                    result = (x, y, w, h)
                    min_dist = dis
        else :
            min_area_diff = 10000000
            for x, y, w, h in f2 :
                area = w * h
                area_diff = abs(area - self.meanBallSize)
                if area_diff < min_area_diff:
                    result = (x, y, w, h)
                    min_area_diff = area_diff

        if len(detected_rects) != 0 and len(merged) != 0 and len(f1) != 0 and len(f2) > 1 and result is not None and self.last_result is not None and False:
            cv2.imshow("inRange", masked)

            # create a black image
            det = np.zeros(frame.shape, np.uint8)
            for x, y, w, h in detected_rects:
                cv2.rectangle(det, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow("detected", det)

            mer = np.zeros(frame.shape, np.uint8)
            for x, y, w, h in merged:
                cv2.rectangle(mer, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.imshow("merged", mer)

            con = np.zeros(frame.shape, np.uint8)
            cv2.polylines(con, [self.consider_poly], True, (0, 255, 255), 4)
            for x, y, w, h in merged:
                cv2.rectangle(con, (x, y), (x + w, y + h), (255, 0, 0), 2)
            for x, y, w, h in f0:
                cv2.rectangle(con, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("considered", con)

            las = np.zeros(frame.shape, np.uint8)
            for m in self.last_map :
                for x, y, w, h in m :
                    cv2.rectangle(las, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imshow("last 20 frames", las)

            #balls = cv2.addWeighted(con, 0.5, las, 0.5, 0)
            balls = np.zeros(frame.shape, np.uint8)
            cv2.polylines(balls, [self.consider_poly], True, (0, 255, 255), 4)
            for x, y, w, h in merged:
                cv2.rectangle(balls, (x, y), (x + w, y + h), (255, 0, 0), 2)
            for x, y, w, h in f0:
                cv2.rectangle(balls, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for m in self.last_map :
                for x, y, w, h in m :
                    cv2.rectangle(balls, (x, y), (x + w, y + h), (0, 0, 255), 2)
            for x, y, w, h in f1:
                cv2.rectangle(balls, (x, y), (x + w, y + h), (0, 128, 255), 3)
            cv2.imshow("balls", balls)

            img = frame.copy()
            cv2.rectangle(img, (result[0], result[1]), (result[0] + result[2], result[1] + result[3]), (0, 128, 255), 3)
            cv2.rectangle(img, (self.last_result[0], self.last_result[1]), (self.last_result[0] + self.last_result[2], self.last_result[1] + self.last_result[3]), (0, 0, 255), 3)
            cv2.polylines(img, [self.consider_poly], True, (0, 255, 255), 4)
            cv2.imshow("result", img)

            # save images for debugging
            cv2.imwrite("results/" + self.save_name + "/frame.jpg", frame)
            cv2.imwrite("results/" + self.save_name + "/inRange.jpg", masked)
            cv2.imwrite("results/" + self.save_name + "/detected.jpg", det)
            cv2.imwrite("results/" + self.save_name + "/merged.jpg", mer)
            cv2.imwrite("results/" + self.save_name + "/considered.jpg", con)
            cv2.imwrite("results/" + self.save_name + "/last.jpg", las)
            cv2.imwrite("results/" + self.save_name + "/balls.jpg", balls)
            cv2.imwrite("results/" + self.save_name + "/result.jpg", img)

            cv2.waitKey(0)
            
        self.last_result = result
        self.last_map.append(f0)
        if len(self.last_map) > 20 :
            self.last_map.pop(0)
        return result

    # start detection.
    def runDetection(self, fromFrameIndex=0, realTime=True, debugging=False) :
        pt = 0
        iteration = 0
        for i in range(fromFrameIndex) :
            self.getNextFrame()
        
        if self.mode == "dual_analysis" or self.mode == "dual_run" or self.mode == "caculate_bounce":
            self.conn.send("ready")
            # wait for main process to send start signal
            while True :
                if self.conn.poll() :
                    msg = self.conn.recv()
                    if msg == "start" :
                        break
        
        self.cam = cv2.VideoCapture(self.source)
        while(True) :
            ret, frame = self.getNextFrame()
            if ret :
                # for caculating iteration time
                this_iter_time = time.perf_counter()

                # save video frame
                if self.mode == "analysis" or self.mode == "dual_analysis":
                    self.video_writer_all.write(frame)
                    pass

                # get detection result
                result = self.findBallInFrame(frame)

                if result is not None :
                    x, y, w, h = result
                    # 在畫面中畫出偵測到的矩形
                    self.drawDirection(frame, x, y, h, w, 1)
                    if self.homography_matrix is not None and self.camera_position is not None:
                        # 單應性變換
                        ball_in_world = np.matmul(self.homography_matrix, np.array([x+w//2, y+h//2, 1]))
                        ball_in_world = ball_in_world / ball_in_world[2]
                        # 獲取投影點
                        projection = equ.Point3d(ball_in_world[0], ball_in_world[1], 0)
                        # 將投影點和相機座標連成直線
                        line = equ.LineEquation3d(self.camera_position, projection)

                        # handle line data
                        if self.mode == "analysis" or self.mode == "dual_analysis":
                            # save result to file
                            self.detection_csv_writer.writerow([iteration, 1, x, y, h, w, self.camera_position.x, self.camera_position.y, self.camera_position.z, line.line_xy.getDeg(), line.line_xz.getDeg()])
                        if self.mode == "compute":
                            # save result to self.data
                            self.data.append([iteration, 1, x, y, h, w, self.camera_position.x, self.camera_position.y, self.camera_position.z, line.line_xy.getDeg(), line.line_xz.getDeg()])
                        if self.mode == "dual_analysis" or self.mode == "dual_run":
                            # send result to main process
                            self.queue.put([self.pid, iteration, self.camera_position.x, self.camera_position.y, self.camera_position.z, line.line_xy.getDeg(), line.line_xz.getDeg(), time.time()])
                            # if bounce detected, send bounce signal to main process
                            if self.bounce_checker.update(y+h//2) :
                                self.conn.send(("bounce", self.pid, iteration))
                        if self.mode == "caculate_bounce" :
                            # save bounce data to self.data
                            if self.bounce_checker.update(y+h//2) :
                                print("bounce")
                                self.data.append(iteration)
                    else :
                        # as above, but this chunk wasn't setup in 3d
                        if self.mode == "analysis" or self.mode == "dual_analysis":
                            self.detection_csv_writer.writerow([iteration, 1, x, y, h, w, 0, 0, 0, 0, 0])
                        elif self.mode == "compute":
                            self.data.append([iteration, 1, x, y, h, w, 0, 0, 0, 0, 0])
                        elif self.mode == "caculate_bounce" :
                            if self.bounce_checker.update(y+h//2) :
                                print("bounce")
                                self.data.append(iteration)
                else :
                    # if no ball detected
                    if self.mode == "analysis" or self.mode == "dual_analysis":
                        # save result to file
                        self.detection_csv_writer.writerow([iteration, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                    if self.mode == "compute":
                        # save result to self.data
                        self.data.append([iteration, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                    if self.mode == "dual_analysis" or self.mode == "dual_run":
                        # send result to main process
                        self.queue.put([self.pid, iteration, None, None, None, None, None, time.time()])
                    

                #save situation data and video frame
                if self.mode == "analysis" or self.mode == "dual_analysis":
                    if result is None :
                        self.video_writer_bad.write(frame)
                    else:
                        self.video_writer_tagged.write(frame)
                    self.video_writer_all_tagged.write(frame)
                pass
                
                # show the detection
                window = "Source" + str(self.source) 
                if self.mode == "analysis" or self.mode == "dual_analysis" or self.mode == "dual_run" or self.mode == "caculate_bounce":
                    cv2.imshow(window, frame)
            else :
                # no frame readed from source. break the loop
                break

            # check if there is data sent from main process
            if self.mode == "dual_analysis" or self.mode == "dual_run" or self.mode == "caculate_bounce":
                if self.conn.poll() :
                    msg = self.conn.recv()
                    # if main process send stop signal, break the loop
                    if msg == "stop" :
                        break

            # check if any key pressed to the window shown
            key = cv2.waitKey(1 if not debugging else 0)
            # if "q" pressed, break the loop
            if key == ord('q') :
                if self.mode == "dual_analysis" or self.mode == "dual_run" or self.mode == "caculate_bounce":
                    self.conn.send("stop")
                break
            # if other key pressed, send the key to main process
            elif key != -1 :
                if self.mode == "dual_analysis" or self.mode == "dual_run":
                    self.conn.send(("keyPress", self.pid, key))
            
            iteration += 1
            pt += time.perf_counter() - this_iter_time


            # wait to match frame rate
            if self.frame_rate != 0 and realTime:
                while time.perf_counter() - this_iter_time < 1/self.frame_rate :
                    pass
            
        if type(self.cam) == CameraReceiver :
            self.cam.close()
        
        if self.mode == "dual_analysis" or self.mode == "dual_run":
            self.conn.send("stop")
        return iteration

# Detection_img is a class for detecting ball in images. It is used for testing.
class Detection_img(Detection) :
    def __init__(self, source, calibrationFile="calibration",frame_size=(640,480), frame_rate=30, color_range="color_range", save_name="default", mode="analysis", beg_ind=0) :
        super().__init__(None, calibrationFile, frame_size, frame_rate, color_range, save_name, mode=mode)
        self.source = source
        self.frameIndex = beg_ind
    def getNextFrame(self):
        img = cv2.imread(os.path.join(self.source, str(self.frameIndex).zfill(4) + ".jpg"))
        if img is None :
            return False, None
        self.frameIndex += 1
        return True, img


_poly = []
_poly_now_pos = (0, 0)
def _setup_poly_mouse_event(event, x, y, flags, param) :
    global _poly, _poly_now_pos
    # check if left button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        _poly.append([x, y])
        print("click at ({}, {})".format(x, y))
    # check if right button was clicked
    elif event == cv2.EVENT_RBUTTONDOWN:
        _poly.pop()
        print("pop")
    elif event == cv2.EVENT_MOUSEMOVE:
        _poly_now_pos = [x, y]


# setup the consider_poly for detection in 2d. return a numpy array of the poly
def setup_poly(source) :
    global _poly, _poly_now_pos
    cam = cv2.VideoCapture(source)
    cv2.namedWindow("setup poly")
    cv2.setMouseCallback("setup poly", _setup_poly_mouse_event)
    while True :
        ret, frame = cam.read()
        if ret :
            k = cv2.waitKey(round(1/30*1000))
            # draw poly
            if len(_poly) > 1 :
                cv2.polylines(frame, np.array([_poly + [_poly_now_pos]]), True, (0, 255, 0), 2)
            cv2.imshow("setup poly", frame)
            if k == ord(' ') :
                break
    cv2.destroyAllWindows()
    ret = np.array(_poly)
    _poly = []
    _poly_now_pos = (0, 0)
    return ret

# this function is no longer used.
def detectProcess(source, save_name) :
    detector = Detection(source=source, save_name=save_name)
    detector.runDetection()
        
# check if two rectangles are overlapped
def check_overlap(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    if x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1:
        return False
    return True

# pad a rectangle with value (make it bigger from all sides)
def pad(rect, value) :
    return (rect[0]-value, rect[1]-value, rect[2] + 2 * value, rect[3] + 2 * value)

# merge rectangles that are overlapped or close to each other, and return the merged rectangles in list
def merge_rectangles(rectangles):
    merged_rectangles = []
    # value is the padding value. It is used to make the rectangles bigger so that they can be merged more easily.
    value=3

    for rect in rectangles:
        padded_rect = pad(rect, value)
        if len(merged_rectangles) == 0:
            merged_rectangles.append(rect)
        else:
            merged = False
            for i, merged_rect in enumerate(merged_rectangles):
                if check_overlap(padded_rect, pad(merged_rect, value)):
                    x = min(rect[0], merged_rect[0])
                    y = min(rect[1], merged_rect[1])
                    w = max(rect[0] + rect[2], merged_rect[0] + merged_rect[2]) - x
                    h = max(rect[1] + rect[3], merged_rect[1] + merged_rect[3]) - y
                    merged_rectangles[i] = (x, y, w, h)
                    merged = True
                    break

            if not merged:
                merged_rectangles.append(rect)

    return merged_rectangles

if __name__ == "__main__" :
    Args = ap.ArgumentParser()
    Args.add_argument("-cc", "--create_config", action="store_true", default=False, help="use this flag to create config, otherwise use config to run detection")
    Args.add_argument("-c", "--config", type=str, default="test1", help="config name (nessesary)")
    Args.add_argument("-s", "--source", type=str, default="all.mp4", help="camera source (nessesary)")
    Args.add_argument("-f", "--frame_size", type=str, default="640x480", help="frame size (only needed when create config, default is 640x480)")
    Args.add_argument("-r", "--frame_rate", type=int, default=30, help="frame rate (only needed when create config, default is 30)")
    Args.add_argument("-cr", "--color_range", type=str, default="cr3", help="color range file name (only needed when create config, default is cr3)")
    Args.add_argument("-i", "--inmtx", type=str, default="calibration", help="camera intrinsic matrix file name (only needed when create config in 3d setup)")
    Args.add_argument("--non_3d_setup", action="store_true", default=False, help="use this flag to setup camera without 3d setup")

    args = Args.parse_args()
    source = args.source if not args.source.isnumeric() else int(args.source)
    # create config for detection
    if args.create_config :
        assert args.source is not None and args.config is not None
        cameraInfo = None
        if not args.non_3d_setup :
            if not os.path.exists(os.path.join("configs", args.inmtx)) :
                raise Exception("intrinsic matrix file not found")
            cameraInfo = setup_table_info(source, load(os.path.join("configs", args.inmtx)))
        createConfig(source, args.config, frame_size=tuple(map(int, args.frame_size.split("x"))), frame_rate=args.frame_rate, color_range=args.color_range, camera_info=cameraInfo)
    # run detection using config
    else :
        assert args.source is not None and args.config is not None
        config = DetectionConfig()
        config.load(args.config)
        dect = Detection(source=source, config=config, save_name=args.config, mode="analysis")
        dect.runDetection(debugging=False, realTime=False)
    

    exit()