import multiprocessing as mp
import os
import sys
sys.path.append(os.getcwd())
import ball_detection.Detection as Detection
import core.common as common
from typing import List, Tuple

class LineCollector:
    lines:List[Tuple[int, int, int, int, int]]
    def put(self, x, y, z, rxy, rxz) :
        if self.checkHit(x, y, z, rxy, rxz) :
            self.lines.clear()
        else :
            if len(self.lines) > 50:
                self.lines.pop(0)
            self.lines.append((x, y, z, rxy, rxz))
    def checkHit(self, x, y, z, rxy, rxz) :
        pass

class LineCollector_hor(LineCollector) :
    movement = None
    last_rxy = None
    def checkHit(self, x, y, z, rxy, rxz):
        if self.movement == 1:
            if rxy - self.last_rxy < 0:
                self.movement = None
                self.last_rxy = None
                return False
        elif self.movement == -1 :
            if rxy - self.last_rxy > 0:
                self.movement = None
                self.last_rxy = None
                return False
        elif self.movement == None and self.last_rxy is not None:
            self.movement = 1 if rxy - self.last_rxy > 0 else -1
        self.last_rxy = rxy
        return True


def predict(
        calibrationFile="calibration",
        frame_size=(640,480), 
        frame_rate=30, 
        color_range="color_range", 
        save_name="dual_default", 
        mode="analysis",
        model=None) :

    # load model
    
    
    common.replaceDir(os.path.join("ball_detection/result/", save_name))
    queue = mp.Queue()
    cam1_pos, cam1_homo = Detection.setup_camera(0, calibrationFile=calibrationFile)
    cam2_pos, cam2_homo = Detection.setup_camera(1, calibrationFile=calibrationFile)
    cam1_kwargs = {
        "calibrationFile" : calibrationFile,
        "frame_size":frame_size, 
        "frame_rate":frame_rate, 
        "color_range":color_range, 
        "save_name":save_name + "/cam1", 
        "mode":"dual_analysis",
        "cam_pos":cam1_pos, 
        "homography_matrix":cam1_homo,
        "queue":queue
    }
    cam2_kwargs = {
        "calibrationFile" : calibrationFile,
        "frame_size":frame_size, 
        "frame_rate":frame_rate, 
        "color_range":color_range, 
        "save_name":save_name + "/cam1", 
        "mode":"dual_analysis",
        "cam_pos":cam2_pos, 
        "homography_matrix":cam2_homo,
        "queue":queue
    }
    p1 = mp.Process(target=Detection.detectProcess, args=(0), kwargs=cam1_kwargs)
    p2 = mp.Process(target=Detection.detectProcess, args=(1), kwargs=cam2_kwargs)
    lines1 = LineCollector_hor()
    lines2 = LineCollector_hor()

    p1.start()
    p2.start()
    try :
        while True :
            new_data = queue.get()
            if new_data[0] == p1.pid:
                lines1.put(new_data[2], new_data[3], new_data[4], new_data[5], new_data[6])
            elif new_data[0] == p2.pid:
                lines2.put(new_data[2], new_data[3], new_data[4], new_data[5], new_data[6])
            
    except KeyboardInterrupt:
        pass

if __name__ == "__main__" :
    predict()
