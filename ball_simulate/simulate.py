import pybullet
import ball_simulate as sim
import core.Equation3d as equ

G = 9.8

class Work:
    timestamp:float
    def getBallPosition(self):
        pass
    def action(self):
        pass

class CameraWork(Work):
    def __init__(self, timestamp, curve, camera_pos, index):
        self.timestamp = timestamp
        self.curve = curve
        self.camera_pos = camera_pos
        self.index = index
    def action(self):
        
        