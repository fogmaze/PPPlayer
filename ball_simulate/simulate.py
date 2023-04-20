import pybullet as p
import os
import sys
sys.path.append(os.getcwd())
import core.Equation3d as equ
import math
import random
import time

G = 9.8

BALL_AREA_HALF_LENGTH = 3
BALL_AREA_HALF_WIDTH = 2
BALL_AREA_HEIGHT = 1

CAMERA_AREA_HALF_LENGTH = 7
CAMERA_AREA_HALF_WIDTH = 5
CAMERA_AREA_HEIGHT = 1.5

class Work:
    timestamp:float
    def action(self,ball_pos):
        pass

class CameraWork(Work):
    def __init__(self, timestamp, camera_pos, index):
        self.timestamp = timestamp
        self.camera_pos = camera_pos
        self.index = index
        
    def action(self, ball_pos):
        lineCamBall = equ.LineEquation3d(self.camera_pos, ball_pos)
        self.rad_xy = math.atan(lineCamBall.line_xy)
        self.rad_xz = math.atan(lineCamBall.line_xz)

class BallWork(Work):
    def __init__(self, timestamp, index):
        self.timestamp = timestamp
        self.index = index

    def action(self, ball_pos):
        self.ball_pos = ball_pos

def randomCameraPos():
    x = 0
    y = 0
    z = 0
    # if pos not in ball_area, return random pos
    while x < BALL_AREA_HALF_LENGTH and x > -BALL_AREA_HALF_LENGTH and y < BALL_AREA_HALF_WIDTH and y > -BALL_AREA_HALF_WIDTH and z < BALL_AREA_HEIGHT:
        x = random.uniform(-CAMERA_AREA_HALF_LENGTH, CAMERA_AREA_HALF_LENGTH)
        y = random.uniform(-CAMERA_AREA_HALF_WIDTH, CAMERA_AREA_HALF_WIDTH)
        z = random.uniform(0, CAMERA_AREA_HEIGHT)
    return equ.Point3d(x, y, z)

def randomBallPos():
    x = random.uniform(-BALL_AREA_HALF_LENGTH, BALL_AREA_HALF_LENGTH)
    y = random.uniform(-BALL_AREA_HALF_WIDTH, BALL_AREA_HALF_WIDTH)
    z = random.uniform(0, BALL_AREA_HEIGHT)
    return equ.Point3d(x, y, z)


def simulate():
    # 设置GUI模式，如果不需要可注释掉此行
    p.connect(p.GUI)

    # 创建平面
    planeId = p.createCollisionShape(p.GEOM_PLANE)
    plane = p.createMultiBody(0, planeId)

    # 创建球
    radius = 0.04
    sphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    startPos = [0, 0, 1]
    startOrientation = p.getQuaternionFromEuler([0,0,0])
    sphere = p.createMultiBody(27, sphereId, basePosition=startPos, baseOrientation=startOrientation)

    # 随机设置球的速度
    linearVelocity = [random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(0, 5)]
    angularVelocity = [0, 0, 0]
    p.resetBaseVelocity(sphere, linearVelocity, angularVelocity)

    # 添加空气阻力
    linearDamping = 6 * math.pi * radius * 0.0185
    angularDamping = 0.1
    p.changeDynamics(sphere, -1, linearDamping=linearDamping, angularDamping=angularDamping)

    # 添加彈性
    restitution = 0.9 # 彈性係數
    p.changeDynamics(sphere, -1, restitution=restitution)
    p.changeDynamics(plane, -1, restitution=restitution)


    # 模拟自由落体过程
    p.setGravity(0, 0, -G)
    while True:
        cam_pos = randomCameraPos()
        ball_pos = randomBallPos()
        p.stepSimulation()
        time.sleep(1/240)


if __name__ == "__main__":
    simulate()