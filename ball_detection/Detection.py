import time
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
import core.common as common
from camera_calibrate.utils import *
from core.Constants import *
import core.Equation3d as equ
import camera_calibrate.utils as utils
import camera_calibrate.Calibration as calib
from camera_reciever.CameraReceiver import CameraReceiver
import time


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

def getBallProjectionPoint(homography_matrix, frame_size, x, y, w, h) :
    ball_in_world = np.matmul(homography_matrix, np.array([frame_size[0] - (x+w//2), y+h//2, 1]))
    projection = equ.Point3d(ball_in_world[0], 0, ball_in_world[1])
    return projection

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
                calibrationFile    = "calibration",
                frame_size         = (640,480), 
                frame_rate         = 30, 
                color_range        = "color_range", 
                save_name          = "default", 
                mode               ="analysis", 
                cam_pos            = None, 
                homography_matrix  = None,
                queue              = None,
                conn               = None
                ) :

        self.pid = os.getpid()
        self.frame_size = frame_size
        self.frame_rate = frame_rate
        if type(cam_pos) == np.ndarray :
            self.camera_position = equ.Point3d(cam_pos[0], cam_pos[1], cam_pos[2])
        else :
            self.camera_position = cam_pos
        if type(color_range) == str :
            self.range = load(color_range)
        else :
            self.range = color_range
        self.upper = self.range.upper
        self.lower = self.range.lower
        self.homography_matrix = homography_matrix
        self.video_writer_all = None
        self.video_writer_bad = None
        self.video_writer_tagged = None
        self.inmtx = calib.load_calibration(calibrationFile)
        self.cam = None
        self.source = source
        self.data = []
        self.conn = None
        if self.frame_size == (640,480) :
            self.meanBallSize = self.MEAN_BALL_SIZE_DICT["320"]
        elif self.frame_size == (1920,1080) :
            self.meanBallSize = self.MEAN_BALL_SIZE_DICT["1080"]
        elif self.frame_size == (1280,720) :
            self.meanBallSize = self.MEAN_BALL_SIZE_DICT["720"] ################################
        else :
            raise Exception("frame size is not supported")

        if type(source) == str and source.replace(".", "").isdigit() :
            self.cam = CameraReceiver(source)
        elif type(source) == str or type(source) == int:
            self.cam = cv2.VideoCapture(source)
        elif type(source) == CameraReceiver :
            self.cam = source

        self.mode = mode
        if self.mode == "analysis" or self.mode == "dual_analysis":
            common.replaceDir("ball_detection/result", save_name)
            self.video_writer_all = cv2.VideoWriter("ball_detection/result/" + save_name + "/all.mp4", self.fourcc, self.frame_rate, self.frame_size)
            self.video_writer_bad = cv2.VideoWriter("ball_detection/result/" + save_name + "/bad.mp4", self.fourcc, self.frame_rate, self.frame_size)
            self.video_writer_tagged = cv2.VideoWriter("ball_detection/result/" + save_name + "/tagged.mp4", self.fourcc, self.frame_rate, self.frame_size)
            self.detection_csv = open("ball_detection/result/" + save_name + "/detection.csv", "w", newline='')
            self.situation_csv = open("ball_detection/result/" + save_name + "/situation.csv", "w", newline='')
            self.detection_csv_writer = csv.writer(self.detection_csv)
            self.situation_csv_writer = csv.writer(self.situation_csv)
            self.detection_csv_writer.writerow(["iter", "id", "x", "y", "h", "w", "cam_x", "cam_y", "cam_z","rxy", "rxz"])
            self.situation_csv_writer.writerow(["iter", "time", "fps", "numOfBall"])
            # pickle camera position and homography matrix
            if self.camera_position is not None and self.homography_matrix is not None:
                with open("ball_detection/result/" + save_name + "/camera_position", "wb") as f :
                    pickle.dump(self.camera_position, f)
                with open("ball_detection/result/" + save_name + "/homography_matrix", "wb") as f :
                    pickle.dump(self.homography_matrix, f)
        if self.mode == "dual_analysis" or self.mode == "dual_run":
            if queue is None or cam_pos is None or homography_matrix is None:
                raise Exception("dual_analysis mode need pipe, cam_pos and homography_matrix")
            self.queue = queue
            self.conn = conn

    def __del__(self) :
        if self.mode == "analysis" or self.mode == "dual_analysis":
            if self.video_writer_all is not None :
                self.video_writer_all.release()
            if self.video_writer_bad is not None :
                self.video_writer_bad.release()
            if self.video_writer_tagged is not None :
                self.video_writer_tagged.release()
            if self.detection_csv is not None :
                self.detection_csv.close()
            if self.situation_csv is not None :
                self.situation_csv.close()
        cv2.destroyAllWindows()

            
    def drawDirection(self, frame, x, y, h, w, i) :
        xCenter = x + w // 2
        yCenter = y + h // 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (xCenter, yCenter), 2, (0, 0, 255), -1)
        cv2.putText(frame, ("x : {} y : {}".format(xCenter, yCenter)), (10, 40*i), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

    def compareFrames(self, frame, compare) :
        return frame
        move = cv2.bitwise_xor(frame, compare)
        color = cv2.inRange(move, np.array([10, 10, 10]), np.array([255, 255, 255]))
        return cv2.bitwise_and(frame, cv2.cvtColor(color, cv2.COLOR_GRAY2BGR))
    
    def maskFrames(self, frame) :
        return cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), self.lower, self.upper)

    def detectContours(self, frame) :
        return cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

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

    def getNextFrame(self) :
        return self.cam.read()
   
    def runDetection(self, img=None, fromFrameIndex=0, realTime=True, debugging=False) :
        BG_CONSIDERED_FRAMES = 60
        MAX_BALL_DISTANCE = 100
        whetherTheFirstFrame = True
        startTime = time.perf_counter()
        last_iter_time = startTime
        last_map = []
        last_result = None
        pt = 0
        iteration = 0
        if img is not None :
            self.camera_position = utils.calculateCameraPosition(self.inmtx, img)
            self.homography_matrix = find_homography_matrix_to_apriltag(img)
        for i in range(fromFrameIndex) :
            self.getNextFrame()
        
        if self.mode == "dual_analysis" or self.mode == "dual_run" :
            self.conn.send("ready")
            # wait for main process to send start signal
            while True :
                if self.conn.poll() :
                    msg = self.conn.recv()
                    if msg == "start" :
                        break
        
        if type(self.cam) == CameraReceiver :
            self.cam.connect()
        while(True) :
            ret, frame = self.getNextFrame()
            if ret :
                this_iter_time = time.perf_counter()

                if self.mode == "analysis" or self.mode == "dual_analysis":
                    self.video_writer_all.write(frame)
                    pass

                if whetherTheFirstFrame :
                    compare = frame
                    whetherTheFirstFrame = False
                
                c = self.compareFrames(frame, compare)
                m = self.maskFrames(c)
                detected = self.detectContours(m)
                # draw contours

                if debugging:
                    cv2.drawContours(frame, detected, -1, (0, 255, 255), 2)
                    cv2.drawContours(c, detected, -1, (0, 255, 255), 2)
                    #frame = cv2.hconcat([frame, m])
                    cv2.imshow("mask", m)
                qualified = []
                for contour in detected :
                    #area = cv2.contourArea(contour)
                    x, y, w, h = cv2.boundingRect(contour)
                    if True :# self.isBallFeature(area, h, w) :
                        qualified.append((x, y, w, h))
                merged = merge_rectangles(qualified)

                f1 = []
                for x, y, w, h in merged :
                    q = True
                    for l in last_map:
                        for x1, y1, w1, h1 in l :
                            if abs(x - x1) < 5 and abs(y - y1) < 5 :
                                q = False
                                break
                    if q :
                        f1.append((x, y, w, h))

                last_map.append(merged)
                if len(last_map) > 20 :
                    last_map.pop(0)

                f2 = []
                for x, y, w, h in f1 :
                    if self.meanBallSize * 0.05 < w * h < self.meanBallSize * 2:
                        f2.append((x, y, w, h))
                result = None
                if last_result is not None :
                    min_dist = 10000000
                    for x, y, w, h in f2 :
                        dis = math.sqrt(abs(x - last_result[0])**2 + abs(y - last_result[1])**2)
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

                last_result = result
                if result is not None :
                    x, y, w, h = result
                    self.drawDirection(frame, x, y, h, w, 1)
                    if self.homography_matrix is not None and self.camera_position is not None:
                        ball_in_world = np.matmul(self.homography_matrix, np.array([x+w//2, y+h//2, 1]))
                        projection = equ.Point3d(ball_in_world[0], 0, ball_in_world[1])
                        line = equ.LineEquation3d(self.camera_position, projection)

                        # save line data
                        if self.mode == "analysis" or self.mode == "dual_analysis":
                            self.detection_csv_writer.writerow([iteration, 1, x, y, h, w, self.camera_position.x, self.camera_position.y, self.camera_position.z, line.line_xy.getDeg(), line.line_xz.getDeg()])
                        elif self.mode == "compute":
                            self.data.append([iteration, 1, x, y, h, w, self.camera_position.x, self.camera_position.y, self.camera_position.z, line.line_xy.getDeg(), line.line_xz.getDeg()])
                        if self.mode == "dual_analysis" or self.mode == "dual_run":
                            self.queue.put([self.pid, iteration, self.camera_position.x, self.camera_position.y, self.camera_position.z, line.line_xy.getDeg(), line.line_xz.getDeg(), time.time()])
                    else :
                        # save pos data
                        if self.mode == "analysis" or self.mode == "dual_analysis":
                            self.detection_csv_writer.writerow([iteration, 1, x, y, h, w, 0, 0, 0, 0, 0])
                        elif self.mode == "compute":
                            self.data.append([iteration, 1, x, y, h, w, 0, 0, 0, 0, 0])

                #save situation data and video
                if self.mode == "analysis" or self.mode == "dual_analysis":
                    self.situation_csv_writer.writerow([iteration, this_iter_time - startTime, 1/(this_iter_time-last_iter_time), 1 if len(merged) > 0 else 0])
                    if result is None :
                        self.video_writer_bad.write(frame)
                    else:
                        self.video_writer_tagged.write(frame)
                
                window = "Source" + str(self.source) 
                if self.mode == "analysis" or self.mode == "dual_analysis" or self.mode == "dual_run":
                    cv2.imshow(window, frame)
            else :
                # send stop signal to main process
                if self.mode == "dual_analysis" or self.mode == "dual_run":
                    print("send stop signal")
                    self.conn.send("stop")
                break
            # check conn from main process
            if self.mode == "dual_analysis" or self.mode == "dual_run":
                if self.conn.poll() :
                    msg = self.conn.recv()
                    if msg == "stop" :
                        print("stop signal received")
                        break
            key = cv2.waitKey(1 if not debugging else 0)
            if key == ord('q') :
                if self.mode == "dual_analysis" or self.mode == "dual_run":
                    self.conn.send("stop")
                break
            iteration += 1
            # print("pid : {} process time : {}".format(self.pid, time.perf_counter() - this_iter_time))
            pt += time.perf_counter() - this_iter_time


            # wait to match frame rate
            if self.frame_rate != 0 and realTime:
                while time.perf_counter() - this_iter_time < 1/self.frame_rate :
                    pass
            
            last_iter_time = this_iter_time
        
        if type(self.cam) == CameraReceiver :
            self.cam.close()
        
        
        if self.mode == "dual_analysis" or self.mode == "dual_run":
            self.conn.send("stop")

        #print("pid : {} avg process time : {} ".format(self.pid, pt/iteration))
        return iteration

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

def setup_camera(source, calibrationFile="calibration") :
    pos = None
    homo = None
    cam = cv2.VideoCapture(source)
    while True :
        ret, frame = cam.read()
        if ret :
            k = cv2.waitKey(1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("W", gray)
            if k == ord(' ') :
                pos = utils.calculateCameraPosition(calib.load_calibration(calibrationFile), gray)
                homo = find_homography_matrix_to_apriltag(gray)
                print("camera pos")
                print(pos.to_str())
                break
    return pos, homo

def setup_camera_img(img_gray, calibrationFile) :
    pos = utils.calculateCameraPosition(calib.load_calibration(calibrationFile), img_gray)
    if pos is None :
        raise Exception("pos not found")
    homo = find_homography_matrix_to_apriltag(img_gray)
    return pos, homo


def setup_camera_android(source, calibrationFile="calibration") :
    pos = None
    homo = None
    isCam = True
    if type(source) == str :
        cam = CameraReceiver(source)
    elif type(source) == np.ndarray :
        frame = source
        ret = True
        isCam = False
    else :
        cam = source
    
    if isCam :
        cam.connect()
    while True :
        if isCam :
            ret, frame = cam.read()
        
        if ret :
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("setup camera", gray)
            k = cv2.waitKey(1)
            if k == ord(' ') :
                pos = utils.calculateCameraPosition(calib.load_calibration(calibrationFile), gray)
                if pos is None :
                    print("pos not find")
                    continue
                homo = find_homography_matrix_to_apriltag(gray)
                print("camera pos")
                print(pos.to_str())
                break
    if isCam :
        cam.close()
    cv2.imwrite("t{}.jpg".format(time.time()), frame)
    
    return pos, homo


def detectProcess(source, save_name) :
    detector = Detection(source=source, save_name=save_name)
    detector.runDetection()
        
def check_overlap(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    if x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1:
        return False
    return True

def pad(rect, value) :
    return (rect[0]-value, rect[1]-value, rect[2] + 2 * value, rect[3] + 2 * value)

def merge_rectangles(rectangles):
    merged_rectangles = []
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
    ini = cv2.imread("exp/t1696229110.0360625.jpg", cv2.IMREAD_GRAYSCALE)
    pos, ho = setup_camera_img(ini, "calibration1")
    dect = Detection(source="exp/3.mp4", color_range="color_range_2", frame_size=(640, 480), save_name="c1",cam_pos=pos, homography_matrix=ho, mode="analysis" )
    dect.runDetection(debugging=False, realTime=False)


    exit()
    detector1 = Detection()
    detector2 = Detection()

    camera1 = mp.Process(target=detectProcess, args=(0, "camera1"))
    camera2 = mp.Process(target=detectProcess, args=(1, "camera2"))
    
    camera1.start()
    camera2.start()
    
    while True :
        try:
            if not camera1.is_alive() or not camera2.is_alive() :
                break
            time.sleep(1)
        except KeyboardInterrupt:
            break
    if camera1.is_alive() :
        camera1.terminate()
    if camera2.is_alive() :
        camera2.terminate()
    camera1.join()
    camera2.join()
    #img = cv2.imread("ball_detection/apriltag-pad.jpg")
    #result = ketstone_correction("ball_detection/apriltag-pad.jpg")
    #src_point = np.float32([[result[0], result[1]], [result[2], result[3]], [result[4], result[5]], [result[6], result[7]]])
    #dst_point = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])
    #perspective_matrix = cv2.getPerspectiveTransform(src_point, dst_point)
    #warped = cv2.warpPerspective(img, perspective_matrix, (640, 480))
    #cv2.imshow("warped", warped)
    #cv2.waitKey(0