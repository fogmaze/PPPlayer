import time
import torch
import csv
import multiprocessing as mp
import os
import sys
import pickle
from typing import List, Tuple
sys.path.append(os.getcwd())
import core.Constants as Constants
import ball_detection.Detection as Detection
from ball_simulate_v2.models import MODEL_MAP
import ball_simulate_v2.models as models
import core.common as common
from ball_detection.ColorRange import *
from camera_reciever.CameraReceiver import CameraReceiver
from ball_predection.predict import LineCollector_hor, prepareModelInput, getHitPointInformation
from robot_controll.controller import Robot

def main(
        robot_ip,
        model_name:str,
        mode,
        weight,
        calibrationFile = "calibration",
        source          = (0, 1),
        frame_size      = (640,480), 
        frame_rate      = 30, 
        color_range     = "color_range", 
        save_name       = "dual_default", 
        ) :

    def runDec(s, sub_name, cp, h, q, c2s) :
        dec = Detection.Detection(
            source = s,
            calibrationFile = calibrationFile,
            frame_size = frame_size, 
            frame_rate = frame_rate, 
            color_range = color_range, 
            save_name = save_name + sub_name, 
            mode = "dual_analysis",# dual_main
            cam_pos = cp, 
            homography_matrix = h,
            queue = q,
            conn=c2s
        )
        dec.runDetection()

    SPEED_UP = False
    HITTING_TIME = 0.1
    RBcontroller = Robot(robot_ip)
    # load model
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
    PREDICT_T = torch.arange(0, Constants.SIMULATE_TEST_LEN * Constants.CURVE_SHOWING_GAP, Constants.CURVE_SHOWING_GAP).to("cuda:0").view(1, -1)

    model:models.ISEFWINNER_BASE = MODEL_MAP[model_name](device="cuda:0")
    model.cuda()
    model.load_state_dict(torch.load(weight))
    model.eval()
    
    common.replaceDir("ball_detection/result/", save_name)
    pf = open("ball_detection/result/" + save_name + "/pred.csv", "w")
    pfw = csv.writer(pf)
    pfw.writerow(["which camera", "frame", "x", "y", "z"])

    queue = mp.Queue()
    c12d, c12s = mp.Pipe()
    c22d, c22s = mp.Pipe()
    if type(source) == tuple and len(source) == 2 :
        if type(source[0]) == int :
            cam1_pos, cam1_homo = Detection.setup_camera(source[0], calibrationFile=calibrationFile)
            cam2_pos, cam2_homo = Detection.setup_camera(source[1], calibrationFile=calibrationFile)
            source1 = source[0]
            source2 = source[1]
        else :
            source1 = CameraReceiver(source[0])
            source2 = CameraReceiver(source[1])
            cam1_pos, cam1_homo = Detection.setup_camera_android(source[0], calibrationFile=calibrationFile)
            cam2_pos, cam2_homo = Detection.setup_camera_android(source[1], calibrationFile=calibrationFile)
    else :
        raise Exception("source type error")

    p1 = mp.Process(target=runDec, args=(source1, "/cam1", cam1_pos, cam1_homo, queue, c12s))
    p2 = mp.Process(target=runDec, args=(source2, "/cam2", cam2_pos, cam2_homo, queue, c22s))

    lines1 = LineCollector_hor()
    lines2 = LineCollector_hor()

    process_time = 0
    process_time_iter = 0
    tra_time = 0
    tra_iter = 0

    p1.start()
    p2.start()

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

    try :
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
                    out:torch.Tensor = model(l, l_len, r, r_len, PREDICT_T).view(-1)
                    Constants.normer.unnorm_ans_tensor(out)

                    hitPoint, t = getHitPointInformation(out)
                    if hitPoint is not None :
                        RBcontroller.move(hitPoint[1], hitPoint[2])
                        if abs(t-time.time()) < HITTING_TIME :
                            RBcontroller.hit()
                    
                    process_time += time.time() - nowT
                    process_time_iter += 1
                    pfw.writerow([which, new_data[1]] + out.tolist())
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
    except KeyboardInterrupt:
        c12d.send("stop")
        c22d.send("stop")
    
    p1.join()
    p2.join()
    process_time /= process_time_iter
    print("mean process time: ", process_time, "; it/s: ", process_time_iter / process_time)
    print("mean tra time: ", tra_time / tra_iter, "; it/s: ", tra_iter / tra_time)
    pf.close()


if __name__ == "__main__" :
    main("medium", "normalB", "ball_simulate_v2/model_saves/normalB/epoch_29/weight.pt")
