import cv2
import multiprocessing as mp
import numpy as np
from typing import List, Tuple, Dict, Any
import sys
import os
sys.path.append(os.getcwd())
from ball_detection.Detection import Detection, DetectionConfig

def _sync_process(source, config:DetectionConfig, s) :
    d = Detection(source, config=config, conn=s)

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

    pass