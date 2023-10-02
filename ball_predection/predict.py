import time
import torch
import csv
import multiprocessing as mp
import pickle
from typing import List, Tuple
import os
import sys
sys.path.append(os.getcwd())
import core.Constants as Constants
import ball_detection.Detection as Detection
from ball_simulate_v2.models import MODEL_MAP
import ball_simulate_v2.models as models
import core.common as common
from ball_detection.ColorRange import *
from camera_reciever.CameraReceiver import CameraReceiver
import core.Equation3d as equ

def getHitPointInformation(_traj_unnormed:torch.Tensor) :
    traj = _traj_unnormed.view(-1, 3)
    if not traj[0][0] < 2.74 < traj[-1][0] :
        return None, None
    l = 0
    r = traj.shape[0] - 1
    while l < r :
        mid = (l + r) // 2
        if traj[mid][0] < 2.74 :
            l = mid + 1
        else :
            r = mid
    l = l - 1
    w = (2.74 - traj[l][0]) / (traj[l+1][0] - traj[l][0])
    hit_point = traj[l] * (1 - w) + traj[l+1] * w
    return hit_point, (l * (1 - w) + (l+1) * w) * Constants.CURVE_SHOWING_GAP

class LineCollector:
    bounceTimestamp = 0
    lines:List[Tuple[int, int, int, int, int]] = []
    def put(self, x, y, z, rxy, rxz) :
        if self.checkHit(x, y, z, rxy, rxz) :
            self.lines.clear()
            return False
        else :
            if len(self.lines) == 0 :
                self.bounceTimestamp = time.time()
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
    l[:len(ll)] = torch.tensor(ll, device=device)
    r[:len(rl)] = torch.tensor(rl, device=device)
    l = l.view(1, Constants.SIMULATE_INPUT_LEN, Constants.MODEL_INPUT_SIZE)
    r = r.view(1, Constants.SIMULATE_INPUT_LEN, Constants.MODEL_INPUT_SIZE)
    l_len = torch.tensor([len(ll)]).view(1,1).to(device)
    r_len = torch.tensor([len(rl)]).view(1,1).to(device)
    return l, l_len, r, r_len

def runDec(s, sub_name, cp, h, q, c2s, calibrationFile, frame_size, frame_rate, color_range, save_name) :
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
        queue = q,
        conn=c2s
    )
    print("start")
    
    dec.runDetection()
    
    
def predict(
        model_name:str,
        weight,
        calibrationFiles = "calibration",
        source          = (0, 1),
        frame_size      = (1280, 720), 
        frame_rate      = 30, 
        color_ranges    = "color_range", 
        save_name       = "dual_default", 
        mode            = "normalB",
        visualization   = True,
        initial_frames  = None
        ) :

    if type(calibrationFiles) == str :
        calibrationFile = (calibrationFiles, calibrationFiles)
    else :
        calibrationFile = calibrationFiles
    if type(color_ranges) == str :
        color_range = (color_ranges, color_ranges)
    else :
        color_range = color_ranges

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
    PREDICT_T = torch.arange(0, Constants.SIMULATE_TEST_LEN * Constants.CURVE_SHOWING_GAP, Constants.CURVE_SHOWING_GAP).to("cuda:0").view(1, -1)
    
    common.replaceDir("ball_detection/result/", save_name)
    pf = open("ball_detection/result/" + save_name + "/pred.csv", "w")
    pfw = csv.writer(pf)
    pfw.writerow(["which camera", "frame", "hp_x", "hp_y", "hp_z", "hp_t", "x", "y", "z"])

    queue = mp.Queue()
    c12d, c12s = mp.Pipe()
    c22d, c22s = mp.Pipe()


    if type(source) == tuple and len(source) == 2 :
        if type(source[0]) == int :
            cam1_pos, cam1_homo = Detection.setup_camera(source[0], calibrationFile=calibrationFile[0])
            cam2_pos, cam2_homo = Detection.setup_camera(source[1], calibrationFile=calibrationFile[1])
            
            source1 = source[0]
            source2 = source[1]
        else :
            source1 = CameraReceiver(source[0])
            source2 = CameraReceiver(source[1])
            if initial_frames is None :
                cam1_pos, cam1_homo = Detection.setup_camera_android(source[0], calibrationFile=calibrationFile[0])
                cam2_pos, cam2_homo = Detection.setup_camera_android(source[1], calibrationFile=calibrationFile[1])
            else :
                #cam1_pos, cam1_homo = Detection.setup_camera_android(initial_frames[0], calibrationFile=calibrationFile[0])
                #cam2_pos, cam2_homo = Detection.setup_camera_android(initial_frames[1], calibrationFile=calibrationFile[1])
                cam1_pos = equ.Point3d(1,1,1)
                cam2_pos = equ.Point3d(1,1,1)
                cam1_homo =  np.array([[1,2,3],[4,5,6],[7,8,9]])
                cam2_homo =  np.array([[1,2,3],[4,5,6],[7,8,9]])
            save("cam1_pos", cam1_pos)
            save("cam1_homo", cam1_homo)
            save("cam2_pos", cam2_pos)
            save("cam2_pos", cam2_pos)
    elif type(source) == str :
        source1 = os.path.join("ball_detection/result", source + "/cam1/all.mp4")
        source2 = os.path.join("ball_detection/result", source + "/cam2/all.mp4")
        with open(os.path.join("ball_detection/result", source + "/cam1/camera_position"), "rb") as f:
            cam1_pos = pickle.load(f)
        with open(os.path.join("ball_detection/result", source + "/cam2/camera_position"), "rb") as f:
            cam2_pos = pickle.load(f)
        with open(os.path.join("ball_detection/result", source + "/cam1/homography_matrix"), "rb") as f:
            cam1_homo = pickle.load(f)
        with open(os.path.join("ball_detection/result", source + "/cam2/homography_matrix"), "rb") as f:
            cam2_homo = pickle.load(f)


    p1 = mp.Process(target=runDec, args=(source1, "/cam1", cam1_pos, cam1_homo, queue, c12s, calibrationFile[0], frame_size, frame_rate, color_range[0], save_name))
    p2 = mp.Process(target=runDec, args=(source2, "/cam2", cam2_pos, cam2_homo, queue, c22s, calibrationFile[1], frame_size, frame_rate, color_range[1], save_name))

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
                which = 1
            elif new_data[0] == p2.pid:
                isHit = not lines2.put(new_data[2], new_data[3], new_data[4], new_data[5], new_data[6])
                which = 2
            # send to model
            if isHit:
                pass
            elif SPEED_UP :
                pass
            else :
                model.reset_hidden_cell(1)
                l, l_len, r, r_len = prepareModelInput(lines1.lines, lines2.lines)
                out:torch.Tensor = model(l, l_len, r, r_len, PREDICT_T) 
                Constants.normer.unnorm_ans_tensor(out)
                hp, t = getHitPointInformation(out)
                print("hit point:", hp, "time:", t)
                process_time += time.time() - nowT
                process_time_iter += 1
                if hp is not None :
                    pfw.writerow([which, new_data[1], hp[0], hp[1], hp[2], t] + out.tolist())
                else :
                    pfw.writerow([which, new_data[1], -1, -1, -1, -1] + out.tolist())
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
        import ball_predection.display as display
        display.visualizePrediction(os.path.join("ball_detection/result", save_name), fps=frame_rate)

if __name__ == "__main__" :
    ini = (cv2.imread("t1696227897.5635931.jpg"), cv2.imread("t1696227897.5635931.jpg"))
    predict("medium", "ball_simulate_v2/model_saves/normalB/epoch_29/weight.pt", calibrationFiles=("calibration_hd", "calibration_hd"), color_ranges="cr_a50", source=("172.20.10.2", "172.20.10.4"), visualization=True, initial_frames=ini)
    #predict("medium", "ball_simulate_v2/model_saves/predict/epoch_29/weight.pt", source=(0, 1))