import pybullet
import test
import core.Equation3d as equ
import math

G = 9.8

class Work:
    timestamp:float
    def action(self,ball_pos):
        pass

class CameraWork(Work):
    def __init__(self, timestamp, curve, camera_pos, index):
        self.timestamp = timestamp
        self.curve = curve
        self.camera_pos = camera_pos
        self.index = index
        
    def action(self, ball_pos):
        lineCamBall = equ.LineEquation3d(self.camera_pos, ball_pos)
        self.deg_xy = math.atan(lineCamBall.line_xy)

if __name__ == "__main__":
    print(math.atan(1))