import time
import cv2
import numpy as np
import math
import multiprocessing as mp
from ColorRange import *
from pupil_apriltags import Detector
import csv
import sys
sys.path.append(os.getcwd())
import core.common as common
from camera_calibrate.utils import *
from core.Constants import *
import core.Equation3d as equ
import camera_calibrate.utils as utils
import camera_calibrate.Calibration as calib


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

class Detection :
    #save test frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    MEAN_BALL_SIZE_DICT = {
        "320" : 175.5024938405144,
        "1080" : 1207.5162448113356,
    }

    def __init__(self, 
                source, 
                calibrationFile="calibration",
                frame_size=(640,480), 
                frame_rate=30, 
                color_range="color_range", 
                save_name="default", 
                mode="analysis", 
                cam_pos=None, 
                homography_matrix=None,
                queue:mp.Queue=None) :

        self.pid = os.getpid()
        self.frame_size = frame_size
        self.frame_rate = frame_rate
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
        if self.frame_size == (640,480) :
            self.meanBallSize = self.MEAN_BALL_SIZE_DICT["320"]
        elif self.frame_size == (1920,1080) :
            self.meanBallSize = self.MEAN_BALL_SIZE_DICT["1080"]
        else :
            raise Exception("frame size is not supported")
        if source is not None :
            self.cam = cv2.VideoCapture(source)

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
        if self.mode == "dual_analysis":
            if queue is None or cam_pos is None or homography_matrix is None:
                raise Exception("dual_analysis mode need pipe, cam_pos and homography_matrix")
            self.queue = queue

    def __del__(self) :
        if self.mode == "analysis" :
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
        move = cv2.bitwise_xor(frame, compare)
        color = cv2.inRange(move, np.array([50, 50, 50]), np.array([255, 255, 255]))
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
        rmax = 18.589264694482747/2.0
        circle = area / a
        if not rmin < r < rmax:
            return False
        if not math.pi * rmin * rmin < area < math.pi * rmax * rmax:
            return False
        if not 0 < circle < 100 :
            return False
        return True

    def getNextFrame(self) :
        return self.cam.read()
   
    def runDetection(self, img=None, fromFrameIndex=0) :
        whetherTheFirstFrame = True
        startTime = time.perf_counter()
        last_iter_time = startTime
        iteration = 0
        if img is not None :
            self.camera_position = utils.calculateCameraPosition(self.inmtx, img)
            self.homography_matrix = find_homography_matrix_to_apriltag(img)
        for i in range(fromFrameIndex) :
            self.getNextFrame()
        while(True) :
            ret, frame = self.getNextFrame()
            if ret :
                key = cv2.waitKey(1)
                this_iter_time = time.perf_counter()

                if self.mode == "analysis" or self.mode == "dual_analysis":
                    self.video_writer_all.write(frame)

                if whetherTheFirstFrame :
                    compare = frame
                    whetherTheFirstFrame = False
                
                if key == ord(' ') :
                    break

                c = self.compareFrames(frame, compare)
                detected = self.detectContours(self.maskFrames(c))
                qualified = []
                for contour in detected :
                    #area = cv2.contourArea(contour)
                    x, y, w, h = cv2.boundingRect(contour)
                    if True :# self.isBallFeature(area, h, w) :
                        qualified.append((x, y, w, h))
                merged = merge_rectangles(qualified)

                min_Area_diff = 500000
                result = None
                for x, y, w, h in merged :
                    area = w * h
                    if abs(area - self.meanBallSize) < min_Area_diff :
                        min_Area_diff = abs(area - self.meanBallSize)
                        result = (x, y, w, h)
                x, y, w, h = result
                self.drawDirection(frame, x, y, h, w, 1)
                if self.homography_matrix is not None and self.camera_position is not None:
                    ball_in_world = np.matmul(self.homography_matrix, np.array([frame.shape[0] - (x+w//2), y+h//2, 1]))
                    projection = equ.Point3d(ball_in_world[0], 0, ball_in_world[1])
                    line = equ.LineEquation3d(self.camera_position, projection)

                    # save line data
                    if self.mode == "analysis" or self.mode == "dual_analysis":
                        self.detection_csv_writer.writerow([iteration, 1, x, y, h, w, self.camera_position.x, self.camera_position.y, self.camera_position.z, line.line_xy.getDeg(), line.line_xz.getDeg()])
                    elif self.mode == "compute":
                        self.data.append([iteration, 1, x, y, h, w, self.camera_position.x, self.camera_position.y, self.camera_position.z, line.line_xy.getDeg(), line.line_xz.getDeg()])
                    if self.mode == "dual_analysis":
                        self.queue.put([self.pid, iteration, self.camera_position.x, self.camera_position.y, self.camera_position.z, line.line_xy.getDeg(), line.line_xz.getDeg()])
                else :
                    # save pos data
                    if self.mode == "analysis" or self.mode == "dual_analysis":
                        self.detection_csv_writer.writerow([iteration, 1, x, y, h, w, 0, 0, 0, 0, 0])
                    elif self.mode == "compute":
                        self.data.append([iteration, 1, x, y, h, w, 0, 0, 0, 0, 0])

                #save situation data and video
                if self.mode == "analysis" or self.mode == "dual_analysis":
                    self.situation_csv_writer.writerow([iteration, this_iter_time - startTime, 1000/(this_iter_time-last_iter_time), 1 if len(merged) > 0 else 0])
                    if len(merged) == 0 :
                        self.video_writer_bad.write(frame)
                    else:
                        self.video_writer_tagged.write(frame)
                
                window = "Source" + str(self.source) 
                if self.mode == "analysis" or self.mode == "dual_analysis":
                    cv2.imshow(window, frame)
            else :
                break
            iteration += 1
            last_iter_time = this_iter_time


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
            if k == ord(' ') :
                pos = utils.calculateCameraPosition(calib.load_calibration(calibrationFile), gray)
                homo = find_homography_matrix_to_apriltag(gray)
                print("camera pos")
                print(pos.to_str)
                break
    return pos, homo

def test_homography(img) :
    homography_matrix = None
    homography_matrix_inv = None
    while True :

        cv2.imshow("frame", img)
        if homography_matrix is None :
            print("No homography matrix press u to update")
        key = cv2.waitKey(10)
        if key == ord('u') :
            detector = Detector()
            detections = detector.detect(img)
            if len(detections) == 1 :
                homography_matrix = detections[0].homography
                homography_matrix_inv = np.linalg.inv(detections[0].homography)
            #homography_matrix = find_homography_matrix_to_apriltag(img)
            if homography_matrix is None :
                print("No tag detected")
        elif key == ord('t') :
            inp = input("input pixel point : ").split()
            if len(inp) == 2 and homography_matrix is not None:
                pxp = np.float128([float(inp[0]), float(inp[1]), 1])
                pxp = np.matmul(homography_matrix, pxp)
                pxp = np.float64([pxp[0]/pxp[2], pxp[1]/pxp[2]])
                print(pxp)
        elif key == ord('r') :
            inp = input("input pixel point : ").split()
            if len(inp) == 2 and homography_matrix is not None:
                pxp = np.float128([float(inp[0]), float(inp[1]), 1])
                pxp = np.matmul(homography_matrix_inv, pxp)
                pxp = np.float64([pxp[0]/pxp[2], pxp[1]/pxp[2]])
                print(pxp)
        elif key == ord('q') :
            break
    cv2.destroyAllWindows()

def detectProcess(source, save_name) :
    detector = Detection(source=source, save_name=save_name)
    detector.runDetection()
        
def check_overlap(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    if x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1:
        return False
    return True

def merge_rectangles(rectangles):
    merged_rectangles = []

    for rect in rectangles:
        if len(merged_rectangles) == 0:
            merged_rectangles.append(rect)
        else:
            merged = False
            for i, merged_rect in enumerate(merged_rectangles):
                if check_overlap(rect, merged_rect):
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
    img = cv2.imread("718.jpg", cv2.IMREAD_GRAYSCALE)

    dect = Detection_img(source="/home/changer/Downloads/hd_60_tagged/frames", frame_size=(1920, 1080), save_name="hd_60_detection", beg_ind=1000)
    dect.runDetection(img)


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