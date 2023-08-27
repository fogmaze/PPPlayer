import torch
import multiprocessing as mp
import os
import sys
import pickle
from typing import List, Tuple
sys.path.append(os.getcwd())
import core.Constants as Constants
import ball_detection.Detection as Detection
import ball_simulate_v2.train as train
import ball_simulate_v2.models as models
import core.common as common

class LineCollector:
    lines:List[Tuple[int, int, int, int, int]]
    def put(self, x, y, z, rxy, rxz) :
        if self.checkHit(x, y, z, rxy, rxz) :
            self.lines.clear()
            return False
        else :
            if len(self.lines) > 50:
                self.lines.pop(0)
            self.lines.append((x, y, z, rxy, rxz))
            return True
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

def get_cam_pos_and_homo(path) :
    pass

def prepareModelInput(ll, rl) :
    l = torch.zeros(Constants.SIMULATE_INPUT_LEN , Constants.MODEL_INPUT_SIZE)
    r = torch.zeros(Constants.SIMULATE_INPUT_LEN , Constants.MODEL_INPUT_SIZE)
    l[:len(ll)] = ll
    r[:len(rl)] = rl
    l = l.view(1, Constants.SIMULATE_INPUT_LEN, Constants.MODEL_INPUT_SIZE)
    r = r.view(1, Constants.SIMULATE_INPUT_LEN, Constants.MODEL_INPUT_SIZE)
    l_len = torch.tensor([len(ll)]).view(1,1)
    r_len = torch.tensor([len(rl)]).view(1,1)
    return l, l_len, r, r_len

def predict(
        model_name:str,
        weight,
        calibrationFile = "calibration",
        source          = (0, 1),
        frame_size      = (640,480), 
        frame_rate      = 30, 
        color_range     = "color_range", 
        save_name       = "dual_default", 
        mode            = "analysis"
        ) :

    SPEED_UP = False

    # load model
    model:models.ISEFWINNER_BASE = train.MODEL_MAP[model_name](device="cuda:0")
    model.cuda()
    model.load_state_dict(weight)
    model.eval()
    
    common.replaceDir(os.path.join("ball_detection/result/", save_name))
    queue = mp.Queue()
    if type(source) == int :
        cam1_pos, cam1_homo = Detection.setup_camera(0, calibrationFile=calibrationFile)
        cam2_pos, cam2_homo = Detection.setup_camera(1, calibrationFile=calibrationFile)
        source1 = source[0]
        source2 = source[1]
    elif type(source) == str :
        source1 = os.path.join("ball_detection/result", source + "/cam1/all.mp4"),
        source2 = os.path.join("ball_detection/result", source + "/cam2/all.mp4")
        with open(os.path.join("ball_detection/result", source + "/cam1/camera_position"), "rb") as f:
            cam1_pos = pickle.load(f)
        with open(os.path.join("ball_detection/result", source + "/cam2/camera_position"), "rb") as f:
            cam2_pos = pickle.load(f)
        with open(os.path.join("ball_detection/result", source + "/cam1/homography_matrix"), "rb") as f:
            cam1_homo = pickle.load(f)
        with open(os.path.join("ball_detection/result", source + "/cam2/homography_matrix"), "rb") as f:
            cam2_homo = pickle.load(f)
        
    def runDec(s, sub_name, cp, h, q) :
        dec = Detection.Detection(
            source = s,
            calibrationFile = calibrationFile,
            frame_size = frame_size, 
            frame_rate = frame_rate, 
            color_range = color_range, 
            save_name = save_name + sub_name, 
            mode = "dual_analysis",
            cam_pos = cp, 
            homography_matrix = h,
            queue = q
        )
        dec.runDetection()

    p1 = mp.Process(target=runDec, args=(source1, "/cam1", cam1_pos, cam1_homo, queue))
    p2 = mp.Process(target=runDec, args=(source2, "/cam2", cam2_pos, cam2_homo, queue))

    lines1 = LineCollector_hor()
    lines2 = LineCollector_hor()

    p1.start()
    p2.start()
    try :
        while True :
            new_data = queue.get()
            isHit = None
            if new_data[0] == p1.pid:
                isHit = not lines1.put(new_data[2], new_data[3], new_data[4], new_data[5], new_data[6])
            elif new_data[0] == p2.pid:
                isHit = not lines2.put(new_data[2], new_data[3], new_data[4], new_data[5], new_data[6])
            if SPEED_UP :
                pass
            else :
                model.reset_hidden_cell(1)
                l, l_len, r, r_len = prepareModelInput(lines1.lines, lines2.lines)
                pass
    except KeyboardInterrupt:
        pass

if __name__ == "__main__" :
    predict()
