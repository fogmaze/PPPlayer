import cv2
import time
import multiprocessing as mp
import numpy as np
from typing import List, Tuple, Dict, Any
import sys
import os
sys.path.append(os.getcwd())
from ball_detection.Detection import Detection, DetectionConfig

def _sync_process(source, config:DetectionConfig, s) :
    d = Detection(source, config=config, conn=s)
    d.runDetection()
    s.send(d.data)

def _caculateSyncTime(a, b) :
    c = []
    for i,j in ((1, 1), (1, 0), (0, 1), (0, 0)) :
        for k,l in ((1, 1), (1, 0), (0, 1), (0, 0)) :
            c.append((a[i:-j if j == 1 else len(a)], b[k:-l if l == 1 else len(b)]))
    res = 20000
    for ca, cb in c:
        if not len(ca) == len(cb) :
            continue
        a_sum = 0
        b_sum = 0
        for i in ca :
            a_sum += i
        for i in cb :
            b_sum += i
        c_res = (a_sum - b_sum) // len(ca)
        if abs(c_res) < abs(res) :
            res = c_res
    return res




def createPredictionConfig(source:Tuple, detectionConfig:Tuple[DetectionConfig, DetectionConfig]) :
    # create four Pipes for communication between processes
    # 0: detection -> prediction
    # 1: prediction -> detection
    # 2: detection -> prediction
    # 3: prediction -> detection
    detection_to_prediction1, prediction_to_detection1 = mp.Pipe()
    p1 = mp.Process(target=_sync_process, args=(source, detectionConfig, detection_to_prediction1))

    detection_to_prediction2, prediction_to_detection2 = mp.Pipe()
    p2 = mp.Process(target=_sync_process, args=(source, detectionConfig, detection_to_prediction2))

    p1.start()
    p2.start()

    while True :
        if prediction_to_detection1.poll() :
            msg = prediction_to_detection1.recv()
            if msg == "ready":
                break
    while True :
        if prediction_to_detection2.poll() :
            msg = prediction_to_detection2.recv()
            if msg == "ready":
                break
    prediction_to_detection1.send("start")
    prediction_to_detection2.send("start")

    while True :
        if prediction_to_detection1.poll() :
            msg = prediction_to_detection1.recv()
            if msg == "stop":
                prediction_to_detection2.send("stop")
                break
        if prediction_to_detection2.poll() :
            msg = prediction_to_detection2.recv()
            if msg == "stop":
                prediction_to_detection1.send("stop")
                break

    data1 = prediction_to_detection1.recv()
    data2 = prediction_to_detection2.recv()   
    


if __name__ == "__main__" :
    a = [58, 78, 93, 149, 182, 273]
    b = [7, 58, 82, 97, 153, 186]

    print(_caculateSyncTime(a, b))