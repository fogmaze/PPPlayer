import cv2
import numpy as np
from ColorRange import *

cam = cv2.VideoCapture(1)
cr = load("color_range")
    
cr.run(cam)    
save("color_range", cr)
