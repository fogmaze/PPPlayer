import cv2
import multiprocessing as mp
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from ball_detection.Detection import Detection

def _sync_process(source, s) :
    d = Detection(source, )

def sync_cameras(source1, source2) :
    pass