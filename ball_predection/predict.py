import matplotlib.pyplot as plt
import time
import torch
import csv
import multiprocessing as mp
import pickle
from typing import List, Tuple
import os
import sys
sys.path.append(os.getcwd())
import core.display as display
import core.Constants as Constants
import ball_detection.Detection as Detection
from ball_simulate_v2.models import MODEL_MAP
import ball_simulate_v2.models as models
import core.common as common
import camera_calibrate.utils as utils
from ball_detection.ColorRange import *
from camera_reciever.CameraReceiver import CameraReceiver
import core.Equation3d as equ

def getHitPointInformation(_traj_unnormed:torch.Tensor) :
    END = 2.74/2
    traj = _traj_unnormed.view(-1, 3)
    if not traj[0][0] < END < traj[-1][0] :
        return None, None
    l = 0
    r = traj.shape[0] - 1
    while l < r :
        mid = (l + r) // 2
        if traj[mid][0] < END :
            l = mid + 1
        else :
            r = mid
    l = l - 1
    w = (END - traj[l][0]) / (traj[l+1][0] - traj[l][0])
    hit_point = traj[l] * (1 - w) + traj[l+1] * w
    return hit_point, (l * (1 - w) + (l+1) * w) * Constants.CURVE_SHOWING_GAP

class LineCollector:
    def __init__(self) :
        self.bounceTimestamp = 0
        self.lines:List[Tuple[int, int, int, int, int]] = []
    def put(self, x, y, z, rxy, rxz) :
        if self.checkHit(x, y, z, rxy, rxz) :
            self.lines.clear()
            return False
        else :
            if len(self.lines) == 0 :
                self.bounceTimestamp = time.time()
            if len(self.lines) >= Constants.SIMULATE_INPUT_LEN:
                self.lines.pop(0)
            self.lines.append((x, y, z, rxy, rxz))
            return True
    def checkHit(self, x, y, z, rxy, rxz) :
        pass
    def clear(self) :
        self.lines.clear()

class LineCollector_hor(LineCollector) :
    def __init__(self):
        super().__init__()
    movement = None
    last_rxy = None
    def clear(self):
        self.last_rxy = None
        self.movement = None
        super().clear()
    def checkHit(self, x, y, z, rxy, rxz):
        print(rxy, self.last_rxy, self.movement)
        if self.movement == 1:
            if rxy - self.last_rxy < 0:
                self.movement = None
                self.last_rxy = None
                return True
        elif self.movement == -1 :
            if rxy - self.last_rxy > 0:
                self.movement = None
                self.last_rxy = None
                return True
        elif self.movement == None and self.last_rxy is not None:
            self.movement = 1 if rxy - self.last_rxy > 0 else -1
        self.last_rxy = rxy
        return False

def prepareModelInput(ll:list, rl:list, device="cuda:0") :
    l = torch.zeros(Constants.SIMULATE_INPUT_LEN , Constants.MODEL_INPUT_SIZE).to(device)
    r = torch.zeros(Constants.SIMULATE_INPUT_LEN , Constants.MODEL_INPUT_SIZE).to(device)
    if len(ll) > 0 :
        l[:len(ll)] = torch.tensor(ll, device=device)
    if len(rl) > 0 :
        r[:len(rl)] = torch.tensor(rl, device=device) 
    l = l.view(1, Constants.SIMULATE_INPUT_LEN, Constants.MODEL_INPUT_SIZE)
    r = r.view(1, Constants.SIMULATE_INPUT_LEN, Constants.MODEL_INPUT_SIZE)
    l_len = torch.tensor([len(ll)]).view(1,1).to(device)
    r_len = torch.tensor([len(rl)]).view(1,1).to(device)
    Constants.normer.norm_input_tensor(l)
    Constants.normer.norm_input_tensor(r)
    return l, l_len, r, r_len

def runDec(s, sub_name, detection_load, q, c2s, save_name) :
    dec = Detection.Detection(
        source = s,
        save_name = save_name + sub_name, 
        mode = "dual_analysis",
        queue = q,
        conn=c2s,
        load_from_result=detection_load
    )
    print("start")
    
    dec.runDetection()
    
    
def predict(
        model_name:str,
        weight,
        source                   = (0, 1),
        detection_load           = (None, None) ,
        save_name                = "dual_default", 
        mode                     = "normalB",
        visualization            = True
        ) :


    SPEED_UP = False
    if mode != "default":
        if mode == "fit" :
            Constants.set2Fitting()
        elif mode == "ne" :
            Constants.set2NoError()
        elif mode == "prediConstantst" :
            Constants.set2Predict()
        elif mode == "normal" :
            Constants.set2Normal()
        elif mode == "normalB" :
            print('a')
            Constants.set2NormalB()
        elif mode == "normalB60" :
            Constants.set2NormalB60()
        else :
            raise Exception("mode error")

    # load model
    model:models.ISEFWINNER_BASE = MODEL_MAP[model_name](device="cuda:0")
    model.cuda()
    model.load_state_dict(torch.load(weight))
    model.eval()
    Constants.set2Normal()
    NORMED_PREDICT_T = torch.arange(0, Constants.SIMULATE_TEST_LEN * Constants.CURVE_SHOWING_GAP, Constants.CURVE_SHOWING_GAP).to("cuda:0").view(1, -1)
    Constants.normer.norm_t_tensor(NORMED_PREDICT_T)

    
    common.replaceDir("ball_detection/result/", save_name)
    pf = open("ball_detection/result/" + save_name + "/pred.csv", "w")
    pfw = csv.writer(pf)
    pfw.writerow(["which camera", "frame", "hp_x", "hp_y", "hp_z", "hp_t", "x", "y", "z"])

    queue = mp.Queue()
    c12d, c12s = mp.Pipe()
    c22d, c22s = mp.Pipe()


    if type(source) == tuple and len(source) == 2 :
        if type(source[0]) == int or type(source[0]) == str:
            
            source1 = source[0]
            source2 = source[1]
        else :
            source1 = CameraReceiver(source[0])
            source2 = CameraReceiver(source[1])
    elif type(source) == str :
        source1 = os.path.join("ball_detection/result", source + "/cam1/all.mp4")
        source2 = os.path.join("ball_detection/result", source + "/cam2/all.mp4")

    p1 = mp.Process(target=runDec, args=(source1, "/cam1", detection_load[0], queue, c12s, save_name))
    p2 = mp.Process(target=runDec, args=(source2, "/cam2", detection_load[1], queue, c22s, save_name))

    lines1 = LineCollector_hor()
    lines2 = LineCollector_hor()

    process_time = 0
    process_time_iter = 0
    tra_time = 0
    tra_iter = 0

    p2.start()
    p1.start()

    while True :
        if c12d.poll() :
            msg = c12d.recv()
            if msg == "ready":
                break
    while True :
        if c22d.poll() :
            msg = c22d.recv()
            if msg == "ready":
                break
    f, ax = display.createFigRoom()
    c12d.send("start")
    c22d.send("start")

    while True :
        if not queue.empty() :
            new_data = queue.get()
            tra_time += time.time() - new_data[7]
            tra_iter += 1

            nowT = time.time()
            isHit = None
            if new_data[0] == p1.pid:
                isHit = not lines1.put(new_data[2], new_data[3], new_data[4], new_data[5], new_data[6])
                if isHit :
                    lines2.clear()
                which = 1
            elif new_data[0] == p2.pid:
                isHit = not lines2.put(new_data[2], new_data[3], new_data[4], new_data[5], new_data[6])
                if isHit :
                    lines1.clear()
                which = 2
            # send to model
            if isHit:
                pass
            elif SPEED_UP :
                pass
            elif len(lines1.lines) > 1 and len(lines2.lines) > 1 :
                model.reset_hidden_cell(1)
                l, l_len, r, r_len = prepareModelInput(lines1.lines, lines2.lines)
                out:torch.Tensor = model(l, l_len, r, r_len, NORMED_PREDICT_T) 
                Constants.normer.unnorm_ans_tensor(out)
                hp, t = getHitPointInformation(out)
                #print("hit point:", hp, "time:", t)
                process_time += time.time() - nowT
                process_time_iter += 1
                if hp is not None :
                    pfw.writerow([which, new_data[1], hp[0], hp[1], hp[2], t] + out.view(-1).tolist())
                else :
                    pfw.writerow([which, new_data[1], -1, -1, -1, -1] + out.view(-1).tolist())

        if c12d.poll() :
            recv = c12d.recv()
            if recv == "stop" :
                c22d.send("stop")
                break
        if c22d.poll() :
            recv = c22d.recv()
            if recv == "stop" :
                c12d.send("stop")
                break
    c12d.send("stop")
    c22d.send("stop")
    
    p1.join()
    p2.join()
    if not process_time_iter == 0 :
        process_time /= process_time_iter
        print("mean process time:", process_time, "; it/s:", process_time_iter / process_time)
        print("mean tra time:", tra_time / tra_iter, "; it/s:", tra_iter / tra_time)
    pf.close()

    # display
    if visualization :
        display.visualizePrediction(os.path.join("ball_detection/result", save_name), fps=30)

if __name__ == "__main__" :
    #ini = (cv2.imread("exp/t1696229110.0360625.jpg", cv2.IMREAD_GRAYSCALE), cv2.imread("exp/t1696227891.9957368.jpg", cv2.IMREAD_GRAYSCALE))

    #predict("medium", "ball_simulate_v2/model_saves/normalB/epoch_29/weight.pt", frame_size=(640, 480),calibrationFiles_initial=("calibration", "calibration"), calibrationFiles=("calibration_hd", "calibration"), color_ranges="cr3", source=("exp/3.mp4", "exp/4.mp4"), visualization=True, initial_frames_gray=ini)
    predict("medium", "ball_simulate_v2/model_saves/normalB/epoch_29/weight.pt", ("exp/3.mp4", "exp/4.mp4"), ("480p30_mid3", "480p30_r"), visualization=True, )
    #predict("medium", "ball_simulate_v2/model_saves/predict/epoch_29/weight.pt", source=(0, 1))
    exit()
    ini = cv2.imread("exp/t1696227891.9957368.jpg", cv2.IMREAD_GRAYSCALE)
    pos, ho = Detection.setup_camera_img(ini[1], "calibration")
    print(pos.to_str(), ho.shape)
    save("pos", pos)
    save("ho", ho)
    exit()